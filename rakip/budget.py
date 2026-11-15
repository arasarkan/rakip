"""Eğitim rejimi seçimi ve süre bütçesi (torch gerektirmez, yerelde test edilir).

Kaggle seansı 12 saatte kesilir ve haftalık GPU kotası sınırlıdır. Eğitim
başlamadan önce süre tahmin edilir; tahmin bütçeyi aşıyorsa önce epoch sayısı,
sonra veri miktarı (en eski pozisyonlar atılarak) otomatik düşürülür.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import FULL_FINETUNE_GAMES, LOW_DATA_GAMES

TIME_BUDGET_HOURS = 10.0
MIN_EPOCHS = 2


def choose_regime(n_games: int) -> str:
    if n_games < LOW_DATA_GAMES:
        return "embedding"
    if n_games < FULL_FINETUNE_GAMES:
        return "light"
    return "full"


# rejim -> (en çok epoch, gövde öğrenme oranı, oyuncu vektörü öğrenme oranı)
REGIME_DEFAULTS = {
    "embedding": (15, 1e-4, 1e-3),
    "light": (10, 5e-5, 1e-3),
    "full": (6, 2e-5, 5e-4),
}


@dataclass
class BudgetPlan:
    epochs: int
    n_train: int               # kullanılacak eğitim pozisyonu sayısı
    steps_per_epoch: int
    estimated_seconds: float
    reduced: bool
    notes: list[str]

    def describe(self) -> str:
        h = self.estimated_seconds / 3600
        text = (
            f"Tahmini eğitim süresi: {format_duration(self.estimated_seconds)} "
            f"({self.epochs} epoch x {self.steps_per_epoch} adım, {self.n_train} pozisyon)"
        )
        if h < 0.02:
            text += " - çok kısa"
        return "\n".join([text] + [f"  - {n}" for n in self.notes])


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f} sn"
    if seconds < 5400:
        return f"{seconds / 60:.0f} dk"
    return f"{seconds / 3600:.1f} saat"


def plan_budget(
    n_train: int,
    n_val: int,
    batch_size: int,
    sec_per_train_step: float,
    sec_per_eval_step: float,
    max_epochs: int,
    budget_seconds: float = TIME_BUDGET_HOURS * 3600,
    fixed_overhead_seconds: float = 0.0,
) -> BudgetPlan:
    """Bütçeye sığan (epoch, pozisyon sayısı) planını üretir.

    `fixed_overhead_seconds`: eğitim dışındaki işler (Stockfish değerlendirmesi,
    dışa aktarım) için ayrılan süre.
    """

    def estimate(epochs: int, n: int) -> tuple[float, int]:
        steps = max(1, math.ceil(n / batch_size))
        eval_steps = max(1, math.ceil(n_val / batch_size)) if n_val else 0
        return fixed_overhead_seconds + epochs * (steps * sec_per_train_step + eval_steps * sec_per_eval_step), steps

    notes: list[str] = []
    epochs, n = max_epochs, n_train
    total, steps = estimate(epochs, n)
    if total > budget_seconds:
        while epochs > MIN_EPOCHS and total > budget_seconds:
            epochs -= 1
            total, steps = estimate(epochs, n)
        if epochs < max_epochs:
            notes.append(f"Süre bütçesi için epoch sayısı {max_epochs} -> {epochs} düşürüldü.")
    if total > budget_seconds:
        per_position = (total - fixed_overhead_seconds) / max(1, n)
        available = max(0.0, budget_seconds - fixed_overhead_seconds)
        keep = int(available / per_position) if per_position > 0 else n
        keep = max(batch_size, min(n, keep))
        if keep < n:
            notes.append(f"Süre bütçesi için eğitim verisi {n} -> {keep} pozisyona düşürüldü (en yeniler tutuldu).")
            n = keep
            total, steps = estimate(epochs, n)
    return BudgetPlan(
        epochs=epochs, n_train=n, steps_per_epoch=steps, estimated_seconds=total,
        reduced=bool(notes), notes=notes,
    )
