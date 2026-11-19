"""Uçtan uca hat: veri -> bölme -> kitap -> ince ayar -> ölçüm -> rapor -> ONNX -> Hub.

Not defteri yalnızca parametreleri verir ve `run` fonksiyonunu çağırır.
Bu dosya torch gerektirir; yalnızca Kaggle/Colab'da çalışır.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import __version__, env as envmod
from .book import build_book, opening_coverage
from .calibrate import TEMPERATURES, calibrate
from .config import BASE_MODEL_REPO, MIN_PLIES
from .data import CollectConfig, collect
from .engine import choose_rows, engine_pass, ensure_stockfish, stockfish_evaluator
from .features import build_samples, nominal_elo
from .games import save_jsonl
from .http import RateLimitedFetcher
from .report import build_report
from .similarity import compute_scores, error_profile, estimate_ceiling, format_scores_text
from .split import Split, assert_ids_clean, assert_no_leakage, manifest, select_extra_training, temporal_split

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
    # Ek tempolardaki oyunların eğitimdeki toplam ağırlığı, hedef tempoya göre en çok
    # bu oran kadar olur (0.5: ek veri toplamda hedef tempo verisinin yarısı kadar etki eder).
    extra_weight: float = 0.5
    # Hamle çeşitliliği (sıcaklık): doğrulama setinde botun hata oranı oyuncununkine
    # yaklaşacak şekilde otomatik seçilir. Kapalıysa `temperature` kullanılır.
    calibrate_temperature: bool = True
    calib_positions: int = 1500
    calib_topk: int = 6


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
    from .train import (
        TrainConfig, calibrate_elo, candidates, fit, load_base_from_hub, pick_device, predict, run_signature, summarize,
    )

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

    # Ek tempolar: yalnızca doğrulama döneminden önceki oyunlar, düşük ağırlıkla.
    extra_fit, extra_info = select_extra_training(collected.extra_games, split)
    extra_info["tempos"] = dict(Counter(g.tempo for g in extra_fit))
    fit_s = train_s
    if extra_fit and cfg.extra_weight > 0:
        per_game = cfg.extra_weight * min(1.0, len(split.train) / len(extra_fit))
        extra_info["weight_per_game"] = per_game
        combined = sorted(split.train + extra_fit, key=lambda g: g.sort_key)
        train_ids = {g.game_id for g in split.train}
        fit_s = build_samples(combined, book, cfg.fallback_elo,
                              game_weight=lambda g: 1.0 if g.game_id in train_ids else per_game)
        assert_ids_clean(fit_s.game_ids, split.test + split.val, "ek tempo eğitim örnekleri")
        assert_no_leakage(Split(train=combined, val=split.val, test=split.test))
        log(f"[ek veri] {len(extra_fit)} oyun eğitime katıldı {extra_info['tempos']} "
            f"(oyun başı ağırlık {per_game:.3f}; {extra_info['cutoff']} ve sonrası hariç: "
            f"{extra_info['dropped']['after_cutoff']} oyun)")
    else:
        extra_info["weight_per_game"] = 0.0
    split_info["extra_training"] = {k: v for k, v in extra_info.items()}
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
    temps = TEMPERATURES if cfg.calibrate_temperature else None
    generic = predict(model, test_s, device, self_elo=self_elo, use_player=False, temperature=cfg.temperature, seed=1,
                      temperatures=temps)
    calib_rows = choose_rows(val_s.measured, cfg.calib_positions, seed=5)
    gen_cand = candidates(model, val_s, device, calib_rows, self_elo, False, cfg.calib_topk, TEMPERATURES) \
        if cfg.calibrate_temperature else None
    log(f"[taban] genel model (reyting {self_elo}): test isabeti %{100 * summarize(generic, test_s)['top1']:.1f}")

    # 5) İnce ayar -------------------------------------------------------------
    train_cfg = TrainConfig(**cfg.train).resolved(len(split.fit) + len(extra_fit))
    signature = run_signature(split_info, train_cfg, self_elo)
    fit_result = fit(model, fit_s, val_s, train_cfg, device, self_elo, signature, paths.work,
                     store=store, session_start=session_start, log=log)
    bot = predict(model, test_s, device, self_elo=self_elo, use_player=True, temperature=cfg.temperature, seed=2,
                  temperatures=temps)
    bot_cand = candidates(model, val_s, device, calib_rows, self_elo, True, cfg.calib_topk, TEMPERATURES) \
        if cfg.calibrate_temperature else None
    log(f"[bot] ince ayarlı model: test isabeti %{100 * summarize(bot, test_s)['top1']:.1f}")

    # 6) Stockfish: referans, hata profili ve rapor ----------------------------
    sf_best, err_bot, err_generic, details = None, None, None, []
    temp_info = {"bot": {"temperature": cfg.temperature, "calibrated": False, "reason": "kapalı"},
                 "generic": {"temperature": cfg.temperature, "calibrated": False, "reason": "kapalı"}}
    engine_path = ensure_stockfish(log)
    if engine_path is None:
        log("[stockfish] bulunamadı; Stockfish karşılaştırması, hata profili ve sıcaklık ayarı atlanıyor.")
        temp_info["bot"]["reason"] = temp_info["generic"]["reason"] = "Stockfish yok"
    else:
        remaining = train_cfg.time_budget_hours * 3600 - (time.time() - session_start)
        budget = max(300.0, min(remaining - 900.0, 2 * 3600.0))

        if cfg.calibrate_temperature and len(calib_rows):
            temp_info = calibrate_temperatures(
                split.val, val_s, calib_rows, bot_cand, gen_cand, engine_path, cfg.engine_nodes, budget * 0.2, log,
            )
            for name, pred in (("bot", bot), ("generic", generic)):
                chosen = temp_info[name]["temperature"]
                if pred.samples_by_t and chosen in pred.samples_by_t:
                    pred.sample = pred.samples_by_t[chosen]

        test_rows = choose_rows(test_s.measured, cfg.max_engine_positions)
        log(f"[stockfish] test: {len(test_rows)} pozisyon değerlendiriliyor...")
        evaluator = stockfish_evaluator(engine_path, nodes=cfg.engine_nodes, time_budget_seconds=budget * 0.5, log=log)
        test_pass = engine_pass(split.test, test_s, test_rows, evaluator, {"bot": bot.sample, "generic": generic.sample})
        sf_best = test_pass.sf_best
        err_bot = error_profile(test_pass.loss_player[test_rows], test_pass.loss["bot"][test_rows], test_pass.phases[test_rows])
        err_generic = error_profile(test_pass.loss_player[test_rows], test_pass.loss["generic"][test_rows], test_pass.phases[test_rows])
        details.extend(test_pass.details)

        report_rows = choose_rows(train_s.measured, cfg.max_report_positions, seed=3)
        log(f"[stockfish] rapor: {len(report_rows)} pozisyon değerlendiriliyor...")
        evaluator = stockfish_evaluator(engine_path, nodes=cfg.engine_nodes, time_budget_seconds=budget * 0.3, log=log)
        details.extend(engine_pass(split.train, train_s, report_rows, evaluator).details)

    # 7) ONNX ve quantize ------------------------------------------------------
    quant = export_and_choose(model, test_s, self_elo, paths.out, torch_reference=bot, log=log)
    # Yayınlanan skor, yayınlanan dosyanın (seçilen ONNX sürümünün) skorudur.
    measured_rows = np.flatnonzero(test_s.measured)
    deployed = onnx_predict(paths.out / "model.onnx", test_s, self_elo, measured_rows)
    deployed.sample = bot.sample

    # 8) Skorlar ve rapor ------------------------------------------------------
    ceiling = estimate_ceiling(split.fit, book=book)
    scores = compute_scores(
        player=player, tempo=tempo, n_games=len(split.fit),
        target=test_s.target, measured=test_s.measured, groups=test_s.game_index,
        bot=deployed, generic=generic, opening=coverage, ceiling=ceiling,
        stockfish_best=sf_best, error_bot=err_bot, error_generic=err_generic,
        n_test_games=len(split.test), excluded=excluded,
    )
    scores["quantization"] = quant
    scores["temperature"] = temp_info
    scores["generic_elo"] = {"chosen": self_elo, "nominal": nominal, "table": elo_table}
    scores["training"] = {
        "regime": fit_result.regime, "best_epoch": fit_result.best_epoch, "stopped_early": fit_result.stopped_early,
        "history": fit_result.history, "trainable_parameters": fit_result.trainable_parameters,
        "seconds": fit_result.seconds, "plan": fit_result.plan, "resumed_from_epoch": fit_result.resumed_from_epoch,
    }
    scores["training"]["extra_games"] = split_info["extra_training"]
    scores["warnings"] = collected.warnings
    report = build_report(
        player=player, tempo=tempo, games=games, details=details,
        counts_by_tempo=collected.counts_by_tempo, warnings=collected.warnings, sources=collected.stats,
    )
    report["data_cleaning"]["events"] = collected.events

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
        "temperature": temp_info["bot"]["temperature"],
        "temperature_calibrated": bool(temp_info["bot"].get("calibrated")),
        "book": {"file": "book.json", "max_ply": book.max_ply, "min_count": book.min_count},
        "games": {"fit": len(split.fit), "test": len(split.test), "extra": len(extra_fit), "min_plies": MIN_PLIES},
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


def calibrate_temperatures(games, samples, rows, bot_cand, gen_cand, engine_path, nodes, seconds, log=print) -> dict:
    """Doğrulama pozisyonlarında bot ve genel model için sıcaklığı ayrı ayrı seçer.

    Her iki modelin en olası adayları tek bir Stockfish geçişinde değerlendirilir;
    hedef, oyuncunun aynı pozisyonlardaki gerçek ciddi hata oranıdır. Ayrıntı:
    rakip/calibrate.py.
    """
    from .similarity import ERROR_THRESHOLD

    rows = np.asarray(rows)
    n = len(samples)
    move_sets = {}
    for prefix, (idx, _, _) in (("b", bot_cand), ("g", gen_cand)):
        for j in range(idx.shape[1]):
            arr = samples.target.astype(np.int32).copy()
            arr[rows] = idx[:, j]
            move_sets[f"{prefix}{j}"] = arr
    log(f"[sıcaklık] {len(rows)} doğrulama pozisyonunda adaylar Stockfish ile değerlendiriliyor...")
    evaluator = stockfish_evaluator(engine_path, nodes=nodes, time_budget_seconds=seconds, log=log)
    result = engine_pass(games, samples, rows, evaluator, move_sets)
    ok = result.sf_best[rows] >= 0
    player_loss = result.loss_player[rows]
    target = float(np.nanmean(player_loss[ok] >= ERROR_THRESHOLD)) if ok.any() else float("nan")

    def err_matrix(prefix, k):
        cols = []
        for j in range(k):
            loss = result.loss[f"{prefix}{j}"][rows]
            cols.append(np.where(np.isnan(loss), np.nan, (loss >= ERROR_THRESHOLD).astype(np.float64)))
        return np.stack(cols, axis=1)[ok]

    info = {}
    for name, prefix, (idx, probs, tail) in (("bot", "b", bot_cand), ("generic", "g", gen_cand)):
        info[name] = calibrate(TEMPERATURES, probs[ok], tail[ok], err_matrix(prefix, idx.shape[1]), target)
        t = info[name]
        if t.get("calibrated"):
            log(f"[sıcaklık] {name}: {t['temperature']} seçildi (oyuncunun hata oranı %{100 * target:.1f}, "
                f"beklenen %{100 * t['expected_error_rate']:.1f}; sıcaklık 1'de %{100 * (t['expected_error_rate_at_1'] or 0):.1f})")
        else:
            log(f"[sıcaklık] {name}: ayarlanamadı ({t.get('reason')}); {t['temperature']} kullanılıyor.")
    return info
