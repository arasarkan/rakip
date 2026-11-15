"""Stockfish geçişi (sahte değerlendiriciyle) ve rakip raporu testleri."""

import json

import chess
import numpy as np

from rakip import report
from rakip.encoding import index_move, move_index
from rakip.engine import EvalResult, choose_rows, engine_pass
from rakip.features import build_samples
from rakip.similarity import error_profile

from conftest import make_record

PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


def material_cp(board: chess.Board, turn: chess.Color) -> float:
    total = 0
    for piece in board.piece_map().values():
        total += PIECE_VALUES[piece.piece_type] * (1 if piece.color == turn else -1)
    return float(total)


def fake_evaluator(jobs):
    """Tek hamlelik malzeme sayımı: en çok malzeme kazandıran hamle 'en iyi'dir."""
    out = []
    for job in jobs:
        board = chess.Board(job.fen)
        scores = {}
        for move in board.legal_moves:
            board.push(move)
            scores[move.uci()] = material_cp(board, not board.turn)
            board.pop()
        best = max(sorted(scores), key=lambda u: scores[u])
        out.append(EvalResult(key=job.key, best=best, cp_best=scores[best], cp_moves={u: scores[u] for u in job.moves}))
    return out


def setup(n_games=12, plies=60):
    games = [make_record(i, f"2024-01-{i + 1:02d}", color="white" if i % 2 == 0 else "black", plies=plies) for i in range(n_games)]
    for g in games:
        g.time_control = "300+0"
        g.clocks = [max(2.0, 300.0 - 5.2 * (k // 2 + 1)) for k in range(len(g.moves))]
        g.result = [1.0, 0.5, 0.0][int(g.game_id[-1]) % 3]
    return games, build_samples(games)


def test_engine_pass_losses_are_consistent():
    games, samples = setup()
    rows = choose_rows(samples.measured, None)
    best_as_bot = np.zeros(len(samples), dtype=np.int32)
    first = engine_pass(games, samples, rows, fake_evaluator)
    best_as_bot[rows] = first.sf_best[rows]
    result = engine_pass(games, samples, rows, fake_evaluator, {"bot": best_as_bot, "same": samples.target.astype(np.int32)})

    evaluated = result.sf_best >= 0
    assert evaluated.sum() == len(rows)
    assert np.nanmin(result.loss_player[rows]) >= 0.0
    # en iyi hamleyi oynayan botun kaybı sıfırdır
    assert np.allclose(result.loss["bot"][rows], 0.0)
    # rakibin hamlesini oynayan botun kaybı rakibinkiyle aynıdır
    assert np.allclose(result.loss["same"][rows], result.loss_player[rows])
    # rakip en iyi hamleyi oynadıysa kaybı sıfırdır
    hit = rows[result.sf_best[rows] == samples.target[rows]]
    assert hit.size > 0 and np.allclose(result.loss_player[hit], 0.0)
    # rastgele oyunlarda malzeme kaybeden hamleler vardır
    assert (result.loss_player[rows] >= 10).sum() > 0
    assert np.isnan(result.loss_player[~evaluated]).all()


def test_engine_pass_details_describe_real_positions():
    games, samples = setup(n_games=4)
    rows = choose_rows(samples.measured, 40, seed=1)
    assert len(rows) == 40
    result = engine_pass(games, samples, rows, fake_evaluator)
    assert len(result.details) == 40
    for d in result.details:
        board = chess.Board(d["fen"])
        game = games[int(samples.game_index[d["row"]])]
        assert game.moves[d["ply"]] == d["played"]
        assert chess.Move.from_uci(d["played"]) in board.legal_moves
        assert board.san(chess.Move.from_uci(d["best"])) == d["best_san"]
        assert d["phase"] in ("opening", "middlegame", "endgame")
        assert move_index(chess.Move.from_uci(d["best"]), board.turn) == result.sf_best[d["row"]]
        assert index_move(int(samples.target[d["row"]]), board.turn).uci() == d["played"]


def test_engine_pass_survives_missing_evaluations():
    games, samples = setup(n_games=3)
    rows = choose_rows(samples.measured, 20)

    def partial(jobs):
        full = fake_evaluator(jobs)
        return [r if i % 2 == 0 else EvalResult(key=r.key) for i, r in enumerate(full)]

    result = engine_pass(games, samples, rows, partial)
    assert (result.sf_best >= 0).sum() == 10
    assert len(result.details) == 10


def test_error_profile_from_engine_pass_is_100_for_identical_play():
    games, samples = setup()
    rows = choose_rows(samples.measured, None)
    result = engine_pass(games, samples, rows, fake_evaluator, {"bot": samples.target.astype(np.int32)})
    profile = error_profile(result.loss_player[rows], result.loss["bot"][rows], result.phases[rows])
    assert profile["score"] == 100.0


def test_report_is_json_serialisable_and_complete():
    games, samples = setup()
    rows = choose_rows(samples.measured, None)
    details = engine_pass(games, samples, rows, fake_evaluator).details
    data = report.build_report(player="Test Oyuncu", tempo="blitz", games=games, details=details,
                               counts_by_tempo={"blitz": 12}, warnings=["uyarı"])
    text = json.dumps(data, ensure_ascii=False)
    assert json.loads(text)["player"] == "Test Oyuncu"
    assert data["summary"]["games"] == 12
    assert data["summary"]["first_game"] == "2024-01-01" and data["summary"]["last_game"] == "2024-01-12"
    assert data["results"]["white"]["games"] == 6 and data["results"]["black"]["games"] == 6
    assert data["results"]["all"]["wins"] + data["results"]["all"]["draws"] + data["results"]["all"]["losses"] == 12
    assert len(data["openings"]["white"]) <= 5
    assert data["phase_errors"]["available"] and data["phase_errors"]["n"] == len(details)
    for phase in data["phase_errors"]["phases"].values():
        if phase["n"]:
            assert 0 <= phase["blunder_rate"] <= phase["error_rate"] <= 1


def test_time_usage_detects_time_trouble_and_handles_missing_clocks():
    games, _ = setup(n_games=6, plies=120)   # 60 hamle x 5.2 sn > 300 sn: herkes zaman sıkıntısına düşer
    usage = report.time_usage(games)
    assert usage["available"] and usage["games_with_clock"] == 6
    assert usage["time_trouble_rate"] == 1.0
    assert 50 <= usage["median_trouble_move"] <= 56      # 30 sn eşiğinin altına 52. hamle civarında iner
    assert "zaman sıkıntısına" in usage["text"]
    first = usage["per_move_range"][0]
    assert abs(first["share_of_base_per_move"] - 5.2 / 300) < 1e-6

    short, _ = setup(n_games=4, plies=40)    # 20 hamle: süre bol
    assert report.time_usage(short)["time_trouble_rate"] == 0.0

    for g in games:
        g.clocks = []
    no_clock = report.time_usage(games)
    assert no_clock["available"] is False and "saat bilgisi yok" in no_clock["text"]


def test_style_profile_counts_structures():
    board = chess.Board()
    moves = []
    for san in ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O", "d6", "d3", "Be6", "Nc3", "Qd7", "Be3", "O-O-O"]:
        move = board.parse_san(san)
        moves.append(move.uci())
        board.push(move)
    game = make_record(1, "2024-01-01", plies=14)
    game.moves = moves
    style = report.style_profile([game])
    assert style["available"] and style["opposite_castling_rate"] == 1.0
    assert style["early_queen_trade_rate"] == 0.0
    assert style["label"] in ("taktik / keskin", "dengeli", "pozisyonel / sakin")
    assert report.style_profile([])["available"] is False


def test_position_tags_and_weaknesses_with_examples():
    rook_ending = {"fen": "8/5pk1/6p1/8/8/6P1/5PK1/3R3r w - - 0 40", "phase": "endgame", "wp_before": 50.0,
                   "clock": 5.0, "time_control": "300+0"}
    tags = report.position_tags(rook_ending)
    assert "Kale oyun sonları" in tags and "Zaman sıkıntısı" in tags and "Dengeli pozisyonlar" in tags
    winning = {"fen": chess.STARTING_FEN, "phase": "middlegame", "wp_before": 90.0, "clock": None, "time_control": None}
    assert "Üstün pozisyonu kazanca çevirme" in report.position_tags(winning)
    assert "Vezirli orta oyunlar" in report.position_tags(winning)

    def detail(i, fen, phase, loss):
        return {"fen": fen, "phase": phase, "wp_before": 50.0, "clock": None, "time_control": None, "loss": loss,
                "ply": 60, "played": "d1d2", "played_san": "Rd2", "best": "d1d7", "best_san": "Rd7",
                "game_id": f"g{i}", "date": "2024-01-01T10:00:00", "opponent": "x", "site": None}

    details = [detail(i, rook_ending["fen"], "endgame", 30.0 if i % 2 else 0.0) for i in range(40)]
    details += [detail(100 + i, chess.STARTING_FEN, "middlegame", 0.0) for i in range(200)]
    weak = report.weaknesses(details)
    assert weak["available"]
    top = weak["items"][0]
    assert top["type"] == "Kale oyun sonları"
    assert top["error_rate"] == 0.5 and top["n"] == 40
    assert len(top["examples"]) == 3
    assert len({e["fen"] for e in top["examples"]}) == 1 and top["examples"][0]["loss"] == 30.0
    assert top["examples"][0]["move_number"] == 31 and top["examples"][0]["date"] == "2024-01-01"
    assert report.weaknesses(details[:5])["available"] is False
