"""Uçtan uca hat: veri -> bölme -> kitap -> ince ayar -> ölçüm -> rapor -> ONNX -> Hub.

Not defteri yalnızca parametreleri verir ve `run` fonksiyonunu çağırır.
Bu dosya torch gerektirir; yalnızca Kaggle/Colab'da çalışır.
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import __version__, env as envmod
from .book import build_book, opening_coverage
from .config import BASE_MODEL_REPO, MIN_PLIES
from .data import CollectConfig, collect
from .engine import choose_rows, engine_pass, ensure_stockfish, stockfish_evaluator
from .features import build_samples, nominal_elo
from .games import save_jsonl
from .http import RateLimitedFetcher
from .report import build_report
from .similarity import compute_scores, error_profile, estimate_ceiling, format_scores_text
from .split import assert_ids_clean, assert_no_leakage, manifest, temporal_split

MIN_GAMES = 30


@dataclass
class RunConfig:
    collect: CollectConfig
    test_fraction: float = 0.20
    val_fraction: float = 0.10
    book_max_ply: int = 30
    book_min_count: int = 3
    temperature: float = 1.0
    fallback_elo: int = 1900
    engine_nodes: int = 40000
    max_engine_positions: int = 6000     # test setinde Stockfish ile değerlendirilecek pozisyon
    max_report_positions: int = 6000     # rapor için değerlendirilecek pozisyon
    hub_private: bool = False
    upload: bool = True
    shutdown_when_done: bool = True
    train: dict = field(default_factory=dict)   # TrainConfig alanları (regime, max_epochs, ...)


def _write_json(path: Path, data) -> None:
    def default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))

    def clean(o):
        if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o

    text = json.dumps(data, ensure_ascii=False, default=default)
    path.write_text(json.dumps(clean(json.loads(text)), ensure_ascii=False, indent=1), encoding="utf-8")


def run(cfg: RunConfig, session_start: float | None = None, log=print) -> dict:
    import torch

    from .export import export_and_choose, onnx_predict
    from .hub import HubStore, model_card
    from .train import TrainConfig, calibrate_elo, fit, load_base_from_hub, pick_device, predict, run_signature, summarize

    session_start = session_start or time.time()
    env = envmod.detect()
    paths = envmod.paths(env)
    player, tempo = cfg.collect.display_name, cfg.collect.tempo
    log(f"[rakip {__version__}] ortam: {env}, oyuncu: {player}, tempo: {tempo}")

    hf_token = envmod.get_secret("HF_TOKEN", env, required=cfg.upload)
    lichess_token = envmod.get_secret("LICHESS_TOKEN", env, required=False)
    store = HubStore.for_player(player, tempo, hf_token, private=cfg.hub_private, log=log) if cfg.upload else None
    if store:
        log(f"[hub] depo: https://huggingface.co/{store.repo_id}")

    # 1) Veri ---------------------------------------------------------------
    fetcher = RateLimitedFetcher(paths.cache / "http", log=log)
    collected = collect(cfg.collect, fetcher, lichess_token=lichess_token, log=log)
    games = collected.games
    if len(games) < MIN_GAMES:
        raise RuntimeError(
            f"{tempo} tempoda yalnızca {len(games)} temiz oyun bulundu; en az {MIN_GAMES} gerekli. "
            f"Tüm tempolar: {collected.counts_by_tempo}. Başka bir tempo ya da kaynak deneyin."
        )
    save_jsonl(games, paths.work / "games.jsonl")

    # 2) Bölme (test seti eğitimden ÖNCE ayrılır) ----------------------------
    split = temporal_split(games, test_fraction=cfg.test_fraction, val_fraction=cfg.val_fraction)
    split_info = manifest(split)
    log(f"[bölme] eğitim {len(split.train)}, doğrulama {len(split.val)}, test {len(split.test)} oyun "
        f"(test: {split_info['date_span']['test'][0]} .. {split_info['date_span']['test'][1]})")

    # 3) Açılış kitabı ve örnekler -------------------------------------------
    book = build_book(split.fit, split.test, max_ply=cfg.book_max_ply, min_count=cfg.book_min_count)
    coverage = opening_coverage(book, split.test)
    train_s = build_samples(split.train, book, cfg.fallback_elo)
    val_s = build_samples(split.val, book, cfg.fallback_elo)
    test_s = build_samples(split.test, book, cfg.fallback_elo)
    assert_ids_clean(train_s.game_ids + val_s.game_ids, split.test, "eğitim örnekleri")
    assert_no_leakage(split)
    excluded = {
        "in_book": int(test_s.in_book.sum()),
        "forced": int(((test_s.n_legal <= 1) & ~test_s.in_book).sum()),
    }
    log(f"[örnek] eğitim {len(train_s)}, doğrulama {len(val_s)}, test {len(test_s)} pozisyon; "
        f"ölçüme giren test pozisyonu {int(test_s.measured.sum())} "
        f"(kitap nedeniyle çıkarılan {excluded['in_book']}, zorunlu hamle {excluded['forced']})")

    # 4) Temel model ve reyting ayarı ----------------------------------------
    device = pick_device()
    log(f"[model] cihaz: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " - GPU YOK, eğitim yavaş olur"))
    model = load_base_from_hub(hf_token, device)
    nominal = nominal_elo(split.fit, cfg.fallback_elo)
    self_elo, elo_table = calibrate_elo(model, val_s, device, nominal, log=log)
    generic = predict(model, test_s, device, self_elo=self_elo, use_player=False, temperature=cfg.temperature, seed=1)
    log(f"[taban] genel model (reyting {self_elo}): test isabeti %{100 * summarize(generic, test_s)['top1']:.1f}")

    # 5) İnce ayar -------------------------------------------------------------
    train_cfg = TrainConfig(**cfg.train).resolved(len(split.fit))
    signature = run_signature(split_info, train_cfg, self_elo)
    fit_result = fit(model, train_s, val_s, train_cfg, device, self_elo, signature, paths.work,
                     store=store, session_start=session_start, log=log)
    bot = predict(model, test_s, device, self_elo=self_elo, use_player=True, temperature=cfg.temperature, seed=2)
    log(f"[bot] ince ayarlı model: test isabeti %{100 * summarize(bot, test_s)['top1']:.1f}")

    # 6) Stockfish: referans, hata profili ve rapor ----------------------------
    sf_best, err_bot, err_generic, details = None, None, None, []
    engine_path = ensure_stockfish(log)
    if engine_path is None:
        log("[stockfish] bulunamadı; Stockfish karşılaştırması ve hata profili atlanıyor.")
    else:
        remaining = train_cfg.time_budget_hours * 3600 - (time.time() - session_start)
        budget = max(300.0, min(remaining - 900.0, 2 * 3600.0))
        test_rows = choose_rows(test_s.measured, cfg.max_engine_positions)
        log(f"[stockfish] test: {len(test_rows)} pozisyon değerlendiriliyor...")
        evaluator = stockfish_evaluator(engine_path, nodes=cfg.engine_nodes, time_budget_seconds=budget * 0.6, log=log)
        test_pass = engine_pass(split.test, test_s, test_rows, evaluator, {"bot": bot.sample, "generic": generic.sample})
        sf_best = test_pass.sf_best
        err_bot = error_profile(test_pass.loss_player[test_rows], test_pass.loss["bot"][test_rows], test_pass.phases[test_rows])
        err_generic = error_profile(test_pass.loss_player[test_rows], test_pass.loss["generic"][test_rows], test_pass.phases[test_rows])
        details.extend(test_pass.details)

        report_rows = choose_rows(train_s.measured, cfg.max_report_positions, seed=3)
        log(f"[stockfish] rapor: {len(report_rows)} pozisyon değerlendiriliyor...")
        evaluator = stockfish_evaluator(engine_path, nodes=cfg.engine_nodes, time_budget_seconds=budget * 0.4, log=log)
        details.extend(engine_pass(split.train, train_s, report_rows, evaluator).details)

    # 7) ONNX ve quantize ------------------------------------------------------
    quant = export_and_choose(model, test_s, self_elo, paths.out, torch_reference=bot, log=log)
    # Yayınlanan skor, yayınlanan dosyanın (seçilen ONNX sürümünün) skorudur.
    measured_rows = np.flatnonzero(test_s.measured)
    deployed = onnx_predict(paths.out / "model.onnx", test_s, self_elo, measured_rows)
    deployed.sample = bot.sample

    # 8) Skorlar ve rapor ------------------------------------------------------
    ceiling = estimate_ceiling(split.fit)
    scores = compute_scores(
        player=player, tempo=tempo, n_games=len(split.fit),
        target=test_s.target, measured=test_s.measured, groups=test_s.game_index,
        bot=deployed, generic=generic, opening=coverage, ceiling=ceiling,
        stockfish_best=sf_best, error_bot=err_bot, error_generic=err_generic,
        n_test_games=len(split.test), excluded=excluded,
    )
    scores["quantization"] = quant
    scores["generic_elo"] = {"chosen": self_elo, "nominal": nominal, "table": elo_table}
    scores["training"] = {
        "regime": fit_result.regime, "best_epoch": fit_result.best_epoch, "stopped_early": fit_result.stopped_early,
        "history": fit_result.history, "trainable_parameters": fit_result.trainable_parameters,
        "seconds": fit_result.seconds, "plan": fit_result.plan, "resumed_from_epoch": fit_result.resumed_from_epoch,
    }
    scores["warnings"] = collected.warnings
    report = build_report(
        player=player, tempo=tempo, games=games, details=details,
        counts_by_tempo=collected.counts_by_tempo, warnings=collected.warnings, sources=collected.stats,
    )

    opp = [g.opponent_rating for g in games if g.opponent_rating]
    meta = {
        "format": 1,
        "rakip_version": __version__,
        "player": player,
        "tempo": tempo,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "base_model": BASE_MODEL_REPO,
        "model": {"file": "model.onnx", "bytes": quant["bytes"], "sha256": quant["sha256"], "precision": quant["chosen"]},
        "inputs": {"history": [64, 96], "elo_self": "scalar", "elo_oppo": "scalar"},
        "elo_self": self_elo,
        "elo_oppo_default": int(statistics.median(opp)) if opp else self_elo,
        "temperature": cfg.temperature,
        "book": {"file": "book.json", "max_ply": book.max_ply, "min_count": book.min_count},
        "games": {"fit": len(split.fit), "test": len(split.test), "min_plies": MIN_PLIES},
        "main_score": scores["main"]["score"],
        "score_state": scores["main"]["state"],
        "confidence": scores["sub"]["confidence"]["level"],
        "confidence_label": scores["sub"]["confidence"]["label"],
        "last_game": report["summary"]["last_game"],
        "config": {k: v for k, v in asdict(cfg).items() if k != "collect"},
    }

    _write_json(paths.out / "meta.json", meta)
    _write_json(paths.out / "scores.json", scores)
    _write_json(paths.out / "report.json", report)
    _write_json(paths.out / "book.json", book.to_json())
    _write_json(paths.out / "split.json", split_info)
    (paths.out / "README.md").write_text(
        model_card(player, tempo, {"main_score": meta["main_score"], "confidence_label": meta["confidence_label"],
                                   "n_games": len(split.fit)}),
        encoding="utf-8",
    )

    log("\n" + "=" * 72 + "\n" + format_scores_text(scores) + "\n" + "=" * 72)
    if store:
        if store.upload_folder(paths.out, f"rakip: {player} ({tempo}) - skor {meta['main_score']}"):
            log(f"[hub] yüklendi: https://huggingface.co/{store.repo_id}")
            log(f"[hub] siteye eklemek için depo adı: {store.repo_id}")
        else:
            log(f"[hub] YÜKLENEMEDİ. Çıktılar şurada duruyor: {paths.out}")
    log(f"[bitti] toplam süre: {(time.time() - session_start) / 60:.0f} dk")
    return {"repo_id": store.repo_id if store else None, "out_dir": str(paths.out), "scores": scores, "meta": meta}
