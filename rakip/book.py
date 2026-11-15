"""Açılış kitabı: rakibin gerçekten oynadığı açılış ağacı, frekanslarıyla.

Kitap yalnızca eğitim + doğrulama oyunlarından kurulur; test oyunları giremez
(`build_book` bunu `assert_ids_clean` ile zorlar). Bot ilk hamlelerde bu
ağaçtan, rakibin gerçek tercih oranlarıyla örnekler.

Bir pozisyon "kitapta" sayılır, eğer:
- oyunun ilk `max_ply` yarım hamlesi içindeyse ve
- rakip o pozisyona en az `min_count` kez gelmişse.
Benzerlik ölçümünde kitaptaki pozisyonlar hamle uyumundan çıkarılır; çünkü
orada bot modelle değil ezberle oynar ve skoru şişirir.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import chess

from .encoding import position_key
from .games import GameRecord
from .split import assert_ids_clean

DEFAULT_MAX_PLY = 30   # ilk 15 hamle
DEFAULT_MIN_COUNT = 3
OPENING_EVAL_PLIES = 16  # açılış uyumu ilk 8 hamle üzerinden ölçülür


@dataclass
class BookEntry:
    moves: Counter = field(default_factory=Counter)   # uci -> oynanma sayısı
    score: float = 0.0                                 # rakibin bu pozisyondan topladığı puan
    scored: int = 0                                    # sonucu bilinen oyun sayısı

    @property
    def total(self) -> int:
        return sum(self.moves.values())


@dataclass
class OpeningBook:
    entries: dict[str, BookEntry]
    max_ply: int = DEFAULT_MAX_PLY
    min_count: int = DEFAULT_MIN_COUNT
    n_games: int = 0

    def lookup(self, board: chess.Board, ply: int | None = None) -> BookEntry | None:
        """Pozisyon kitaptaysa girdisini, değilse None döner."""
        if ply is not None and ply >= self.max_ply:
            return None
        entry = self.entries.get(position_key(board))
        if entry is None or entry.total < self.min_count:
            return None
        return entry

    def probabilities(self, board: chess.Board, ply: int | None = None) -> dict[str, float] | None:
        entry = self.lookup(board, ply)
        if entry is None:
            return None
        total = entry.total
        return {uci: n / total for uci, n in entry.moves.most_common()}

    def sample(self, board: chess.Board, rng: random.Random, ply: int | None = None) -> chess.Move | None:
        """Rakibin gerçek tercih oranlarıyla bir kitap hamlesi seçer."""
        probs = self.probabilities(board, ply)
        if not probs:
            return None
        ucis = list(probs)
        uci = rng.choices(ucis, weights=[probs[u] for u in ucis], k=1)[0]
        move = chess.Move.from_uci(uci)
        return move if move in board.legal_moves else None

    def to_json(self) -> dict:
        """Site için: yalnızca kitapta sayılan (min_count'u geçen) pozisyonlar."""
        positions = {}
        for key, entry in self.entries.items():
            if entry.total < self.min_count:
                continue
            positions[key] = {
                "n": entry.total,
                "moves": dict(entry.moves.most_common()),
                "score": round(entry.score / entry.scored, 4) if entry.scored else None,
            }
        return {"max_ply": self.max_ply, "min_count": self.min_count, "n_games": self.n_games, "positions": positions}

    @classmethod
    def from_json(cls, data: dict) -> "OpeningBook":
        entries = {}
        for key, item in data["positions"].items():
            entry = BookEntry(moves=Counter(item["moves"]))
            if item.get("score") is not None:
                entry.scored = item["n"]
                entry.score = item["score"] * item["n"]
            entries[key] = entry
        return cls(entries, data["max_ply"], data["min_count"], data.get("n_games", 0))


def build_book(
    fit_games: Iterable[GameRecord],
    test_games: Iterable[GameRecord] = (),
    max_ply: int = DEFAULT_MAX_PLY,
    min_count: int = DEFAULT_MIN_COUNT,
) -> OpeningBook:
    fit_games = list(fit_games)
    assert_ids_clean((g.game_id for g in fit_games), test_games, "açılış kitabı")
    entries: dict[str, BookEntry] = defaultdict(BookEntry)
    for game in fit_games:
        board = chess.Board()
        player_turn = chess.WHITE if game.player_color == "white" else chess.BLACK
        for ply, uci in enumerate(game.moves[:max_ply]):
            move = chess.Move.from_uci(uci)
            if board.turn == player_turn:
                entry = entries[position_key(board)]
                entry.moves[uci] += 1
                if game.result is not None:
                    entry.score += game.result
                    entry.scored += 1
            board.push(move)
    return OpeningBook(dict(entries), max_ply=max_ply, min_count=min_count, n_games=len(fit_games))


def opening_coverage(book: OpeningBook, games: Iterable[GameRecord], plies: int = OPENING_EVAL_PLIES) -> dict:
    """Açılış uyumu: rakibin (hiç görülmemiş) oyunlarının ilk hamlelerinde
    yaptığı seçimlerin yüzde kaçı kitapta bir seçenek olarak var?

    Her oyun eşit ağırlıktadır. Dönen "coverage" 0-1 arasıdır.
    """
    per_game: list[float] = []
    known_total = moves_total = 0
    for game in games:
        board = chess.Board()
        player_turn = chess.WHITE if game.player_color == "white" else chess.BLACK
        known = total = 0
        for ply, uci in enumerate(game.moves[:plies]):
            if board.turn == player_turn:
                total += 1
                entry = book.lookup(board, ply)
                if entry is not None and entry.moves.get(uci, 0) > 0:
                    known += 1
            board.push(chess.Move.from_uci(uci))
        if total:
            per_game.append(known / total)
            known_total += known
            moves_total += total
    return {
        "coverage": sum(per_game) / len(per_game) if per_game else 0.0,
        "games": len(per_game),
        "moves_known": known_total,
        "moves_total": moves_total,
        "plies": plies,
    }


def top_lines(games: Iterable[GameRecord], color: str, depth_plies: int = 8, top: int = 5) -> list[dict]:
    """Rapor için: bir renkle en sık oynanan hatlar ve skor yüzdeleri."""
    groups: dict[tuple[str, ...], list[GameRecord]] = defaultdict(list)
    for game in games:
        if game.player_color == color and len(game.moves) >= depth_plies:
            groups[tuple(game.moves[:depth_plies])].append(game)
    total = sum(len(v) for v in groups.values())
    lines = []
    for moves, members in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:top]:
        board = chess.Board()
        san = []
        for uci in moves:
            move = chess.Move.from_uci(uci)
            san.append(board.san(move))
            board.push(move)
        scored = [g.result for g in members if g.result is not None]
        names = Counter(g.opening for g in members if g.opening)
        lines.append({
            "moves_uci": list(moves),
            "moves_san": san,
            "name": names.most_common(1)[0][0] if names else None,
            "eco": Counter(g.eco for g in members if g.eco).most_common(1)[0][0] if any(g.eco for g in members) else None,
            "games": len(members),
            "share": len(members) / total if total else 0.0,
            "score": sum(scored) / len(scored) if scored else None,
            "wins": sum(1 for r in scored if r == 1.0),
            "draws": sum(1 for r in scored if r == 0.5),
            "losses": sum(1 for r in scored if r == 0.0),
        })
    return lines
