"""web/sample/ icin ORNEK rakip verisi uretir (arayuzu egitim yapmadan gorebilmek icin).

Buradaki her sey sentetiktir: oyunlar rastgele devam ettirilmis kisa acilislardir,
"motor" tek hamlelik malzeme sayimidir ve tahminler yapay olarak uretilir. Sayilar
gercek bir olcum DEGILDIR; site bu sayfayi "ornek veri" uyarisiyla gosterir ve
yalnizca yerelde (localhost) ya da adrese ?ornek eklenince listeler.

    python tools/build_sample_opponent.py
"""

import datetime as dt
import json
import random
from pathlib import Path

import chess
import numpy as np

from rakip.book import build_book, opening_coverage
from rakip.data import data_warnings
from rakip.engine import EvalResult, choose_rows, engine_pass
from rakip.features import build_samples
from rakip.games import GameRecord
from rakip.report import build_report
from rakip.similarity import Predictions, compute_scores, error_profile, estimate_ceiling
from rakip.split import manifest, temporal_split

PLAYER = "Örnek Oyuncu"
TEMPO = "blitz"
WHITE_LINES = [
    (["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"], "Ruy Lopez", "C78", 34),
    (["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6"], "Sicilya Savunması", "B50", 26),
    (["e4", "e6", "d4", "d5", "Nc3", "Nf6", "Bg5", "Be7"], "Fransız Savunması", "C11", 14),
    (["e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Bf5"], "Caro-Kann", "B18", 10),
    (["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7"], "Vezir Gambiti", "D53", 6),
]
BLACK_LINES = [
    (["e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "g6"], "Sicilya: Hızlandırılmış Ejderha", "B34", 30),
    (["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6"], "Şah Hint Savunması", "E70", 28),
    (["e4", "c5", "Nc3", "Nc6", "g3", "g6", "Bg2", "Bg7"], "Kapalı Sicilya", "B25", 12),
    (["c4", "e5", "Nc3", "Nf6", "g3", "d5", "cxd5", "Nxd5"], "İngiliz Açılışı", "A22", 8),
    (["Nf3", "Nf6", "g3", "g6", "Bg2", "Bg7", "O-O", "O-O"], "Réti", "A05", 6),
]
VALUES = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


def material(board, turn):
    return float(sum(VALUES[p.piece_type] * (1 if p.color == turn else -1) for p in board.piece_map().values()))


def toy_evaluator(jobs):
    out = []
    for job in jobs:
        board = chess.Board(job.fen)
        scores = {}
        for move in board.legal_moves:
            board.push(move)
            scores[move.uci()] = material(board, not board.turn)
            board.pop()
        best = max(sorted(scores), key=lambda u: scores[u])
        out.append(EvalResult(key=job.key, best=best, cp_best=scores[best], cp_moves={u: scores[u] for u in job.moves}))
    return out


def make_games(n=312, seed=7):
    rng = random.Random(seed)
    start = dt.date(2023, 1, 5)
    games = []
    for i in range(n):
        color = "white" if i % 2 == 0 else "black"
        lines = WHITE_LINES if color == "white" else BLACK_LINES
        san, name, eco, _ = rng.choices(lines, weights=[l[3] for l in lines])[0]
        board = chess.Board()
        moves = []
        for s in san:
            move = board.parse_san(s)
            moves.append(move.uci())
            board.push(move)
        length = rng.randint(50, 110)
        while len(moves) < length and not board.is_game_over():
            legal = list(board.legal_moves)
            # Sentetik oyuncu: cogunlukla malzeme kaybetmeyen hamleler, ara sira hata.
            captures = [m for m in legal if board.is_capture(m)]
            move = rng.choice(captures) if captures and rng.random() < 0.45 else rng.choice(legal)
            moves.append(move.uci())
            board.push(move)
        date = start + dt.timedelta(days=int(i * 2.1))
        clocks, left = [], [180.0, 180.0]
        for ply in range(len(moves)):
            left[ply % 2] = max(1.5, left[ply % 2] - rng.uniform(0.6, 5.5) * (1.6 if 16 < ply < 50 else 0.8) + 2)
            clocks.append(round(left[ply % 2], 1))
        games.append(GameRecord(
            game_id=f"ornek{i:04d}", source="lichess", player_color=color, tempo=TEMPO,
            date=f"{date.isoformat()}T19:{i % 60:02d}:00", result=rng.choices([1.0, 0.5, 0.0], weights=[46, 14, 40])[0],
            player_rating=2080 + int(40 * np.sin(i / 30)) + rng.randint(-15, 15), opponent_rating=2060 + rng.randint(-120, 120),
            opponent_name="Rakip", time_control="180+2", moves=moves, clocks=clocks, eco=eco, opening=name,
        ))
    return games


def fake_predictions(samples, rate, rng, engine_best=None, engine_share=0.0):
    """Yapay tahmin: `rate` oraninda oyuncunun hamlesi, gerisinde baska bir yasal hamle."""
    n = len(samples)
    top1 = samples.target.astype(np.int32).copy()
    sample = top1.copy()
    legal = samples.legal_mask(slice(None))
    for row in range(n):
        options = np.flatnonzero(legal[row])
        other = int(rng.choice(options))
        if engine_best is not None and engine_best[row] >= 0 and rng.random() < engine_share:
            other = int(engine_best[row])
        if rng.random() >= rate:
            top1[row] = other
        if rng.random() >= rate * 0.8:
            sample[row] = int(rng.choice(options))
    return Predictions(top1=top1, p_target=np.full(n, 0.3, np.float32), p_top1=np.full(n, 0.45, np.float32), sample=sample)


def main():
    rng = np.random.default_rng(11)
    games = make_games()
    split = temporal_split(games)
    book = build_book(split.fit, split.test)
    train_s, test_s = build_samples(split.train, book), build_samples(split.test, book)

    rows = choose_rows(test_s.measured, 2500)
    first = engine_pass(split.test, test_s, rows, toy_evaluator)
    bot = fake_predictions(test_s, 0.57, rng)
    generic = fake_predictions(test_s, 0.50, rng, first.sf_best, 0.3)
    test_pass = engine_pass(split.test, test_s, rows, toy_evaluator, {"bot": bot.sample, "generic": generic.sample})
    err_bot = error_profile(test_pass.loss_player[rows], test_pass.loss["bot"][rows], test_pass.phases[rows])
    err_gen = error_profile(test_pass.loss_player[rows], test_pass.loss["generic"][rows], test_pass.phases[rows])
    details = test_pass.details + engine_pass(split.train, train_s, choose_rows(train_s.measured, 3000, seed=3), toy_evaluator).details

    counts = {"classical": 0, "rapid": 41, "blitz": len(games), "bullet": 1250}
    warnings = data_warnings(TEMPO, counts)
    scores = compute_scores(
        player=PLAYER, tempo=TEMPO, n_games=len(split.fit), target=test_s.target, measured=test_s.measured,
        groups=test_s.game_index, bot=bot, generic=generic, opening=opening_coverage(book, split.test),
        ceiling=estimate_ceiling(split.fit, book=book), stockfish_best=test_pass.sf_best, error_bot=err_bot, error_generic=err_gen,
        n_test_games=len(split.test),
        excluded={"in_book": int(test_s.in_book.sum()), "forced": int(((test_s.n_legal <= 1) & ~test_s.in_book).sum())},
    )
    scores["sample"] = True
    scores["quantization"] = {
        "chosen": "int8", "bytes": 5_600_000,
        "variants": [
            {"name": "fp32", "bytes": 21_000_000, "match": scores["references"]["bot"], "accepted": True},
            {"name": "int8", "bytes": 5_600_000, "match": scores["references"]["bot"] - 0.002, "accepted": True},
            {"name": "fp16", "bytes": 10_800_000, "match": scores["references"]["bot"], "accepted": True},
        ],
    }
    report = build_report(player=PLAYER, tempo=TEMPO, games=games, details=details, counts_by_tempo=counts, warnings=warnings)
    report["sample"] = True
    meta = {
        "format": 1, "sample": True, "player": PLAYER, "tempo": TEMPO,
        # Ornek rakibin kendi modeli yoktur; site oyun icin genel demo modelini kullanir.
        "model": {"file": "model.onnx", "bytes": 0, "sha256": "ornek", "precision": "fp16"},
        "elo_self": 2080, "elo_oppo_default": 2060, "temperature": 1.0,
        "book": {"file": "book.json", "max_ply": book.max_ply, "min_count": book.min_count},
        "games": {"fit": len(split.fit), "test": len(split.test)},
        "main_score": scores["main"]["score"], "score_state": scores["main"]["state"],
        "confidence": scores["sub"]["confidence"]["level"], "confidence_label": scores["sub"]["confidence"]["label"],
        "last_game": report["summary"]["last_game"],
    }

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if o != o else float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    out = Path(__file__).resolve().parent.parent / "web" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("meta", meta), ("scores", scores), ("report", report), ("book", book.to_json()), ("split", manifest(split))):
        (out / f"{name}.json").write_text(json.dumps(clean(data), ensure_ascii=False, indent=1), encoding="utf-8")
    print("sample written:", out, "| main score", scores["main"]["score"], scores["main"]["state"])


if __name__ == "__main__":
    main()
