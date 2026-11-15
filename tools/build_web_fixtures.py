"""Tarayici kodlamasinin Python kodlamasiyla ayni oldugunu dogrulamak icin ornek uretir.

Cikti: web/tests/fixtures.json  (web/tests/index.html bu dosyayi okur)
"""

import json
import random
from pathlib import Path

import chess
import numpy as np

from rakip import encoding as enc


def game(seed, plies, start_fen=None):
    rng = random.Random(seed)
    board = chess.Board(start_fen) if start_fen else chess.Board()
    codes, cases, moves = [], [], []
    for ply in range(plies):
        legal = list(board.legal_moves)
        if not legal:
            break
        codes.append(enc.board_codes(board))
        if ply % 5 == 0 or any(m.promotion for m in legal) or ply < 3:
            tokens = enc.stack_history(np.stack(codes), enc.history_indices(ply)[None, :])[0]
            cases.append({
                "ply": ply,
                "fen": board.fen(),
                "key": enc.position_key(board),
                "history_nonzero": np.flatnonzero(tokens.reshape(-1)).tolist(),
                "legal": {m.uci(): enc.move_index(m, board.turn) for m in legal},
            })
        move = rng.choice(legal)
        moves.append(move.uci())
        board.push(move)
    return {"start_fen": start_fen, "moves": moves, "cases": cases}


def main():
    games = [game(seed, 90) for seed in range(6)]
    games.append(game(100, 12, "1n5k/P7/8/8/8/8/7p/K5N1 w - - 0 1"))
    games.append(game(101, 12, "1n5k/P7/8/8/8/8/7p/K5N1 b - - 0 1"))
    games.append(game(102, 30, "r3k2r/pppq1ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPPQ1PPP/R3K2R w KQkq - 0 9"))
    out = Path(__file__).resolve().parent.parent / "web" / "tests" / "fixtures.json"
    out.write_text(json.dumps({"n_moves": enc.N_MOVES, "games": games}), encoding="utf-8")
    print("fixtures:", out, sum(len(g["cases"]) for g in games), "cases")


if __name__ == "__main__":
    main()
