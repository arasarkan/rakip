"""Hamle çeşitliliği (sıcaklık) kalibrasyonu (yalnızca numpy; yerelde test edilir).

SORUN
=====
Bot hamlesini modelin olasılık dağılımından örnekler. Sıcaklık 1'de bu dağılım
"bu seviyedeki insanların ortalaması" gibidir ve tek bir oyuncudan daha
dağınıktır: ilk gerçek denemelerde bot, oyuncunun 1,6 ile 10 katı arası sık
ciddi hata yaptı (Eric Rosen %6 / bot %10, Carlsen %1 / bot %10).

ÇÖZÜM
=====
Doğrulama setindeki pozisyonlarda (test setine dokunmadan):
  1. Modelin en olası K aday hamlesi Stockfish ile değerlendirilir.
  2. Her aday için "ciddi hata mı" (kazanma şansını en az 10 puan düşürüyor mu)
     belirlenir; oyuncunun gerçek hamlesi için de aynısı yapılır.
  3. Her sıcaklık T için botun BEKLENEN hata oranı hesaplanır:
        E(T) = ortalama_pozisyon( Σ_k p_T(k) · hata(k)  +  kuyruk_T · KUYRUK_HATASI )
     p_T: sıcaklık T'deki olasılıklar (tüm yasal hamleler üzerinden normalize),
     kuyruk_T: ilk K aday dışında kalan olasılık. Değerlendirilmeyen bu hamleler
     temkinli biçimde hata sayılır (KUYRUK_HATASI = 1): düşük olasılıklı hamleler
     çoğunlukla zayıftır ve bu varsayım yalnızca sıcaklığı düşürme yönünde yanılır.
  4. Beklenen hata oranı oyuncununkine en yakın olan T seçilir. Eşitlikte 1'e
     en yakın olan tercih edilir.

Sıcaklık hamle uyumunu (ilk tahmin) değiştirmez; yalnızca botun gerçekte
oynarken ne sıklıkla sapacağını belirler. Seçilen değer sitede botun varsayılan
çeşitlilik ayarı olur ve hata profili bu sıcaklıkla ölçülür. Genel model de aynı
yöntemle ayrıca kalibre edilir ki karşılaştırma adil olsun.
"""

from __future__ import annotations

import numpy as np

TEMPERATURES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4)
TAIL_ERROR = 1.0
MIN_POSITIONS = 200


def expected_error_rates(
    probs: np.ndarray,      # (R, nT, K) adayların olasılıkları
    tail: np.ndarray,       # (R, nT)   aday dışı olasılık
    err: np.ndarray,        # (R, K)    1 = ciddi hata, 0 = değil, nan = değerlendirilemedi
    tail_err: float = TAIL_ERROR,
) -> np.ndarray:
    """Her sıcaklık için botun beklenen ciddi hata oranı: (nT,)."""
    filled = np.where(np.isnan(err), tail_err, err).astype(np.float64)
    per_position = (probs * filled[:, None, :]).sum(axis=-1) + tail * tail_err
    return per_position.mean(axis=0)


def choose_temperature(temperatures, rates, target: float) -> float:
    """Beklenen hata oranı hedefe en yakın sıcaklık; eşitlikte 1'e en yakını."""
    temps = np.asarray(temperatures, dtype=np.float64)
    gaps = np.abs(np.asarray(rates, dtype=np.float64) - target)
    best = gaps.min()
    candidates = temps[np.isclose(gaps, best, rtol=0, atol=1e-9)]
    return float(candidates[np.argmin(np.abs(candidates - 1.0))])


def calibrate(
    temperatures,
    probs: np.ndarray,
    tail: np.ndarray,
    err: np.ndarray,
    target: float,
    fallback: float = 1.0,
) -> dict:
    """Sıcaklığı seçer ve tabloyu döner. Az pozisyon varsa `fallback` kullanılır."""
    n = int(probs.shape[0])
    if n < MIN_POSITIONS or target != target:  # target nan ise
        return {"temperature": float(fallback), "calibrated": False, "positions": n,
                "reason": f"yetersiz veri ({n} pozisyon)" if n < MIN_POSITIONS else "oyuncu hata oranı yok"}
    rates = expected_error_rates(probs, tail, err)
    chosen = choose_temperature(temperatures, rates, target)
    index = list(temperatures).index(chosen)
    at_one = float(rates[list(temperatures).index(1.0)]) if 1.0 in temperatures else None
    return {
        "temperature": chosen,
        "calibrated": True,
        "positions": n,
        "target_error_rate": float(target),
        "expected_error_rate": float(rates[index]),
        "expected_error_rate_at_1": at_one,
        "table": [{"temperature": float(t), "expected_error_rate": float(r)} for t, r in zip(temperatures, rates)],
    }


def softmax_temperatures(logits: np.ndarray, temperatures) -> np.ndarray:
    """(R, M) yasal hamle logitleri -> (R, nT, M) olasılıklar. Testler ve tarayıcı tarafı için referans."""
    out = []
    for t in temperatures:
        z = logits / max(float(t), 1e-3)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        out.append(e / e.sum(axis=1, keepdims=True))
    return np.stack(out, axis=1)
