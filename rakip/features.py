"""Oyunlardan model örnekleri üretir (yalnızca numpy).

Her örnek, hedef oyuncunun hamle yaptığı bir pozisyondur. Bellek için tahta
dizilişleri bir kez saklanır (`codes`), örnekler 8'li geçmiş indeksleriyle
onlara işaret eder; (N, 64, 96) girdi tensörü eğitim sırasında toplu halde
üretilir.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Iterable

import chess
import numpy as np

from .book import OpeningBook
from .encoding import (
    LEGAL_BYTES,
    N_MOVES,
    board_codes,
    history_indices,
    legal_indices,
    move_index,
    stack_history,
    unpack_mask,
)
from .games import GameRecord

# Maia, Lichess reytingleriyle eğitildi. Masa başı (FIDE) reytingleri kabaca
# bu kadar düşüktür; fark seviyeye göre değişir, bu yalnızca başlangıç
# tahminidir. Eğitimden önce en uygun reyting doğrulama setinde ayrıca aranır.
OTB_ELO_OFFSET = 200
DEFAULT_ELO = 1900


@dataclass
class SampleSet:
    codes: np.ndarray       # (P, 64) int8, tüm oyunların tüm pozisyonları
    hist: np.ndarray        # (N, 8) int32, codes içine indeksler
    target: np.ndarray      # (N,) int16, oynanan hamlenin sözlük indeksi
    legal: np.ndarray       # (N, 544) uint8, paketlenmiş yasal hamle maskesi
    n_legal: np.ndarray     # (N,) int16
    self_elo: np.ndarray    # (N,) float32
    oppo_elo: np.ndarray    # (N,) float32
    game_index: np.ndarray  # (N,) int32, game_ids içine indeks
    ply: np.ndarray         # (N,) int16
    in_book: np.ndarray     # (N,) bool, pozisyon açılış kitabında mı
    clock: np.ndarray       # (N,) float32, hamleden önce kalan saniye (bilinmiyorsa nan)
    game_ids: list[str]
    weight: np.ndarray | None = None  # (N,) float32 eğitim ağırlığı; None = hepsi 1

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def tokens(self, rows: np.ndarray | slice) -> np.ndarray:
        """(B, 64, 96) float32 model girdisi."""
        return stack_history(self.codes, self.hist[rows])

    def legal_mask(self, rows: np.ndarray | slice) -> np.ndarray:
        return unpack_mask(self.legal[rows])

    def subset(self, rows: np.ndarray) -> "SampleSet":
        return SampleSet(
            codes=self.codes, hist=self.hist[rows], target=self.target[rows], legal=self.legal[rows],
            n_legal=self.n_legal[rows], self_elo=self.self_elo[rows], oppo_elo=self.oppo_elo[rows],
            game_index=self.game_index[rows], ply=self.ply[rows], in_book=self.in_book[rows],
            clock=self.clock[rows], game_ids=self.game_ids,
            weight=None if self.weight is None else self.weight[rows],
        )

    def weights(self) -> np.ndarray:
        return np.ones(len(self), dtype=np.float32) if self.weight is None else self.weight

    @property
    def measured(self) -> np.ndarray:
        """Benzerlik ölçümüne giren örnekler: kitap dışı ve birden fazla yasal hamle."""
        return (~self.in_book) & (self.n_legal > 1)


def nominal_elo(games: Iterable[GameRecord], fallback: int = DEFAULT_ELO) -> int:
    """Oyuncunun Lichess ölçeğindeki tipik reytingi (oyunların medyanı)."""
    ratings = [
        g.player_rating + (OTB_ELO_OFFSET if g.source in ("twic", "pgn") else 0)
        for g in games
        if g.player_rating
    ]
    return int(statistics.median(ratings)) if ratings else fallback


def build_samples(
    games: Iterable[GameRecord],
    book: OpeningBook | None = None,
    fallback_elo: int = DEFAULT_ELO,
    game_weight: Callable[[GameRecord], float] | None = None,
) -> SampleSet:
    """`game_weight` verilirse her oyunun pozisyonları o ağırlıkla eğitime girer
    (ör. ek tempolardaki oyunlar daha düşük ağırlıkla)."""
    games = list(games)
    player_elo = nominal_elo(games, fallback_elo)
    weights: list[float] = []

    codes: list[np.ndarray] = []
    hist, target, legal, n_legal = [], [], [], []
    self_elo, oppo_elo, game_index, plies, in_book, clock = [], [], [], [], [], []

    for gi, game in enumerate(games):
        offset_elo = OTB_ELO_OFFSET if game.source in ("twic", "pgn") else 0
        me = (game.player_rating + offset_elo) if game.player_rating else player_elo
        opp = (game.opponent_rating + offset_elo) if game.opponent_rating else me
        player_turn = chess.WHITE if game.player_color == "white" else chess.BLACK
        base = len(codes)
        board = chess.Board()
        for ply, uci in enumerate(game.moves):
            codes.append(board_codes(board))
            move = chess.Move.from_uci(uci)
            if board.turn == player_turn:
                indices = legal_indices(board)
                mask = np.zeros(N_MOVES, dtype=np.uint8)
                mask[indices] = 1
                hist.append(history_indices(ply, base))
                target.append(move_index(move, board.turn))
                legal.append(np.packbits(mask))
                n_legal.append(len(indices))
                self_elo.append(me)
                oppo_elo.append(opp)
                game_index.append(gi)
                plies.append(ply)
                in_book.append(book is not None and book.lookup(board, ply) is not None)
                # Hamleden önce kalan süre = oyuncunun bir önceki hamlesinden sonra kalan süre.
                before = game.clocks[ply - 2] if game.clocks and ply >= 2 else None
                clock.append(np.nan if before is None else before)
                weights.append(1.0 if game_weight is None else float(game_weight(game)))
            board.push(move)

    n = len(target)
    return SampleSet(
        codes=np.stack(codes) if codes else np.zeros((0, 64), dtype=np.int8),
        hist=np.stack(hist).astype(np.int32) if n else np.zeros((0, 8), dtype=np.int32),
        target=np.asarray(target, dtype=np.int16),
        legal=np.stack(legal) if n else np.zeros((0, LEGAL_BYTES), dtype=np.uint8),
        n_legal=np.asarray(n_legal, dtype=np.int16),
        self_elo=np.asarray(self_elo, dtype=np.float32),
        oppo_elo=np.asarray(oppo_elo, dtype=np.float32),
        game_index=np.asarray(game_index, dtype=np.int32),
        ply=np.asarray(plies, dtype=np.int16),
        in_book=np.asarray(in_book, dtype=bool),
        clock=np.asarray(clock, dtype=np.float32),
        game_ids=[g.game_id for g in games],
        weight=None if game_weight is None else np.asarray(weights, dtype=np.float32),
    )
