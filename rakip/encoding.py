"""Maia-3 girdi/çıktı kodlaması (yalnızca numpy; torch gerektirmez).

Resmi Maia-3 çıkarım koduyla (CSSLab/maia3, AGPL-3.0) birebir uyumludur:

- Tahta her zaman hamle sırası olan tarafın gözünden kodlanır; sıra siyahtaysa
  tahta dikey çevrilir ve renkler değişir.
- Her kare bir "token"dır: 12 kanal = kendi P,N,B,R,Q,K + rakip P,N,B,R,Q,K.
- Model son 8 pozisyonu görür (en eski önce, güncel pozisyon sonda). Her
  pozisyon KENDİ hamle sırasına göre kodlanır. Oyun 8 yarım hamleden kısaysa
  en eski pozisyon öne doğru tekrarlanır.
- Hamle sözlüğü 4352 girdidir: 64x64 (kaynak, hedef) + 256 terfi
  (kaynak sütun x hedef sütun x {q,r,b,n}, hep 7. yataydan 8. yataya).

Aynı kodlama web/src/lib/encoding.ts içinde de vardır; ikisi birlikte değişmeli.
"""

from __future__ import annotations

import chess
import numpy as np

HISTORY = 8
N_SQUARE_MOVES = 64 * 64
N_MOVES = N_SQUARE_MOVES + 256
PROMO_PIECES = "qrbn"
_PROMO_INDEX = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}
_PROMO_TYPES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
LEGAL_BYTES = N_MOVES // 8  # 544


def board_codes(board: chess.Board) -> np.ndarray:
    """(64,) int8: 0 boş, 1-6 sıradaki tarafın P,N,B,R,Q,K, 7-12 rakibinkiler.

    Kare indeksleri sıradaki tarafın gözündendir (kendi 1. yatayı = 0..7).
    """
    codes = np.zeros(64, dtype=np.int8)
    flip = board.turn == chess.BLACK
    for square, piece in board.piece_map().items():
        idx = chess.square_mirror(square) if flip else square
        own = piece.color == board.turn
        codes[idx] = piece.piece_type + (0 if own else 6)
    return codes


def codes_to_tokens(codes: np.ndarray) -> np.ndarray:
    """(..., 64) kodlar -> (..., 64, 12) float32 one-hot."""
    eye = np.concatenate([np.zeros((1, 12), dtype=np.float32), np.eye(12, dtype=np.float32)])
    return eye[codes.astype(np.int64)]


def history_indices(ply: int, offset: int = 0) -> np.ndarray:
    """`ply` pozisyonu için 8 geçmiş pozisyonun indeksleri (en eski önce).

    Oyunun başında eksik kalan yerler en eski pozisyonla (0) doldurulur.
    """
    idx = np.arange(ply - HISTORY + 1, ply + 1)
    return np.maximum(idx, 0) + offset


def stack_history(codes: np.ndarray, hist: np.ndarray) -> np.ndarray:
    """codes: (P, 64) tüm pozisyonlar; hist: (N, 8) indeksler -> (N, 64, 96) float32."""
    tokens = codes_to_tokens(codes[hist])            # (N, 8, 64, 12)
    tokens = np.transpose(tokens, (0, 2, 1, 3))      # (N, 64, 8, 12)
    return np.ascontiguousarray(tokens.reshape(tokens.shape[0], 64, HISTORY * 12))


def move_index(move: chess.Move, turn: chess.Color) -> int:
    """Hamlenin sözlükteki indeksi (sıradaki tarafın gözünden)."""
    src, dst = move.from_square, move.to_square
    if turn == chess.BLACK:
        src, dst = chess.square_mirror(src), chess.square_mirror(dst)
    if move.promotion:
        return N_SQUARE_MOVES + (chess.square_file(src) * 8 + chess.square_file(dst)) * 4 + _PROMO_INDEX[move.promotion]
    return src * 64 + dst


def index_move(index: int, turn: chess.Color) -> chess.Move:
    """Sözlük indeksinden hamle (gerçek tahta koordinatlarında)."""
    if index >= N_SQUARE_MOVES:
        rest = index - N_SQUARE_MOVES
        pair, piece = divmod(rest, 4)
        file_from, file_to = divmod(pair, 8)
        src, dst, promo = chess.square(file_from, 6), chess.square(file_to, 7), _PROMO_TYPES[piece]
    else:
        src, dst = divmod(index, 64)
        promo = None
    if turn == chess.BLACK:
        src, dst = chess.square_mirror(src), chess.square_mirror(dst)
    return chess.Move(src, dst, promotion=promo)


def legal_indices(board: chess.Board) -> list[int]:
    return [move_index(m, board.turn) for m in board.legal_moves]


def legal_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(N_MOVES, dtype=bool)
    mask[legal_indices(board)] = True
    return mask


def pack_mask(mask: np.ndarray) -> np.ndarray:
    return np.packbits(mask.astype(np.uint8), axis=-1)


def unpack_mask(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(packed, axis=-1, count=N_MOVES).astype(bool)


def position_key(board: chess.Board) -> str:
    """Açılış kitabı anahtarı: taş dizilişi + sıra + rok hakları.

    Geçerken alma karesi bilerek dışarıda: farklı kütüphaneler (python-chess,
    chess.js) onu farklı yazar ve kitap iki tarafta da aynı anahtarı üretmeli.
    """
    return " ".join(board.fen(en_passant="fen").split(" ")[:3])
