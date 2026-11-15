"""Zamana göre eğitim / doğrulama / test bölmesi ve sızıntı kontrolleri.

Test seti, oyuncunun EN YENİ oyunlarıdır ve eğitimden önce ayrılır. Böylece
ölçülen şey "model geçmişi ezberledi mi" değil, "geleceği tahmin edebiliyor
mu" olur. Test oyunları eğitime, doğrulamaya ya da açılış kitabına giremez;
`assert_no_leakage` ve `assert_ids_clean` bunu her aşamada zorlar.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

from .games import GameRecord

MIN_TEST_FRACTION = 0.15
MAX_TEST_FRACTION = 0.20


class LeakageError(AssertionError):
    """Test verisi eğitim tarafına sızdı."""


@dataclass
class Split:
    train: list[GameRecord]
    val: list[GameRecord]
    test: list[GameRecord]

    @property
    def fit(self) -> list[GameRecord]:
        """Modelin ve açılış kitabının görebileceği her şey (test hariç)."""
        return self.train + self.val


def temporal_split(
    games: Iterable[GameRecord],
    test_fraction: float = 0.20,
    val_fraction: float = 0.10,
) -> Split:
    """Oyunları tarihe göre sıralar; en yeni dilimi teste, ondan önceki dilimi
    doğrulamaya (erken durdurma için) ayırır.

    Tarihi bilinmeyen oyunlar en eski sayılır ve teste giremez.
    """
    if not MIN_TEST_FRACTION <= test_fraction <= MAX_TEST_FRACTION:
        raise ValueError(f"test_fraction {MIN_TEST_FRACTION}-{MAX_TEST_FRACTION} aralığında olmalı")
    if not 0.0 <= val_fraction < 0.5:
        raise ValueError("val_fraction 0 ile 0.5 arasında olmalı")

    ordered = sorted(games, key=lambda g: g.sort_key)
    n = len(ordered)
    if n < 3:
        raise ValueError("Bölme için en az 3 oyun gerekli")

    dated = sum(1 for g in ordered if g.date)
    n_test = min(max(1, math.ceil(n * test_fraction)), dated)
    if n_test == 0:
        raise ValueError("Hiçbir oyunun tarihi yok; en yeni oyunlar belirlenemediği için test seti ayrılamıyor")
    rest, test = ordered[: n - n_test], ordered[n - n_test:]
    n_val = min(len(rest) - 1, math.ceil(len(rest) * val_fraction)) if val_fraction > 0 else 0
    n_val = max(0, n_val)
    train, val = rest[: len(rest) - n_val], rest[len(rest) - n_val:]

    split = Split(train=train, val=val, test=test)
    assert_no_leakage(split)
    return split


def _moves_key(game: GameRecord) -> str:
    return " ".join(game.moves)


def assert_no_leakage(split: Split) -> None:
    """Bölmenin temiz olduğunu doğrular; değilse LeakageError fırlatır.

    Kontroller:
    1. Aynı oyun kimliği birden fazla dilimde yok.
    2. Test oyunlarından biriyle hamle hamle aynı olan bir oyun eğitim
       tarafında yok (aynı oyunun başka kaynaktan gelmiş kopyası).
    3. Test oyunlarının tamamı tarihli ve hiçbir eğitim/doğrulama oyunu
       en eski test oyunundan daha yeni değil.
    """
    train_ids = {g.game_id for g in split.train}
    val_ids = {g.game_id for g in split.val}
    test_ids = {g.game_id for g in split.test}
    for name, a, b in (
        ("eğitim-test", train_ids, test_ids),
        ("doğrulama-test", val_ids, test_ids),
        ("eğitim-doğrulama", train_ids, val_ids),
    ):
        common = a & b
        if common:
            raise LeakageError(f"{name} arasında ortak {len(common)} oyun var, ör. {sorted(common)[:3]}")

    test_moves = {_moves_key(g) for g in split.test}
    for game in split.fit:
        if _moves_key(game) in test_moves:
            raise LeakageError(f"{game.game_id} oyunu bir test oyunuyla hamle hamle aynı")

    if split.test:
        if any(not g.date for g in split.test):
            raise LeakageError("Test setinde tarihi bilinmeyen oyun var")
        oldest_test = min(g.date[:10] for g in split.test)
        for game in split.fit:
            if game.date and game.date[:10] > oldest_test:
                raise LeakageError(
                    f"{game.game_id} ({game.date}) en eski test oyunundan ({oldest_test}) daha yeni"
                )


def assert_ids_clean(used_game_ids: Iterable[str], test_games: Iterable[GameRecord], where: str) -> None:
    """Bir aşamanın (eğitim, kitap, rapor istatistiği) kullandığı oyunların
    hiçbirinin test setinde olmadığını doğrular."""
    test_ids = {g.game_id for g in test_games}
    leaked = test_ids.intersection(used_game_ids)
    if leaked:
        raise LeakageError(f"{where}: {len(leaked)} test oyunu kullanılmış, ör. {sorted(leaked)[:3]}")


def manifest(split: Split) -> dict:
    """Bölmenin Hub'a kaydedilen özeti; sonradan doğrulama için."""

    def digest(games: list[GameRecord]) -> str:
        return hashlib.sha256("\n".join(sorted(g.game_id for g in games)).encode()).hexdigest()[:16]

    def span(games: list[GameRecord]) -> list[str | None]:
        dates = [g.date[:10] for g in games if g.date]
        return [min(dates), max(dates)] if dates else [None, None]

    return {
        "method": "temporal",
        "counts": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "test_fraction": round(len(split.test) / max(1, len(split.train) + len(split.val) + len(split.test)), 4),
        "date_span": {"train": span(split.train), "val": span(split.val), "test": span(split.test)},
        "digest": {"train": digest(split.train), "val": digest(split.val), "test": digest(split.test)},
        "test_game_ids": sorted(g.game_id for g in split.test),
    }
