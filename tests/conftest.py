"""Testler için sentetik oyun üreticileri. Ağ erişimi yoktur."""

from __future__ import annotations

import random

import chess
import chess.pgn
import pytest

from rakip.games import GameRecord


def random_moves(rng: random.Random, plies: int) -> list[chess.Move]:
    board = chess.Board()
    moves = []
    for _ in range(plies):
        legal = list(board.legal_moves)
        if not legal:
            break
        move = rng.choice(legal)
        board.push(move)
        moves.append(move)
    return moves


def make_pgn(
    white="hedef",
    black="rakip1",
    date="2024.01.15",
    tc="300+3",
    result="1-0",
    plies=30,
    seed=0,
    clocks=True,
    extra_headers: dict | None = None,
    utc=True,
) -> str:
    rng = random.Random(seed)
    game = chess.pgn.Game()
    game.headers["Event"] = "Rated Blitz game"
    game.headers["Site"] = f"https://lichess.org/{seed:08d}"
    game.headers["White"] = white
    game.headers["Black"] = black
    game.headers["Result"] = result
    game.headers["Date"] = date
    if utc:
        game.headers["UTCDate"] = date
        game.headers["UTCTime"] = "12:00:00"
    game.headers["WhiteElo"] = "2100"
    game.headers["BlackElo"] = "2050"
    if tc is not None:
        game.headers["TimeControl"] = tc
    for key, value in (extra_headers or {}).items():
        game.headers[key] = value
    node = game
    remaining = [300.0, 300.0]
    for i, move in enumerate(random_moves(rng, plies)):
        node = node.add_variation(move)
        if clocks:
            remaining[i % 2] -= rng.uniform(1, 8)
            node.set_clock(max(1.0, remaining[i % 2]))
    return str(game)


def make_record(i: int, date: str | None, color="white", plies=24, tempo="blitz") -> GameRecord:
    rng = random.Random(1000 + i)
    moves = [m.uci() for m in random_moves(rng, plies)]
    return GameRecord(
        game_id=f"g{i:05d}",
        source="lichess",
        player_color=color,
        tempo=tempo,
        date=date,
        result=1.0,
        player_rating=2100,
        opponent_rating=2050,
        opponent_name=f"opp{i}",
        time_control="300+3",
        moves=moves,
    )


@pytest.fixture
def dated_records() -> list[GameRecord]:
    # 100 oyun, her biri farklı günde: 2023-01-01'den itibaren
    import datetime as dt

    start = dt.date(2023, 1, 1)
    return [make_record(i, (start + dt.timedelta(days=i)).isoformat()) for i in range(100)]
