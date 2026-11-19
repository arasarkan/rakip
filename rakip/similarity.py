"""Rakip benzerlik skoru (torch gerektirmez; birim testleri yerelde çalışır).

NEYİ ÖLÇÜYORUZ
==============
Bot, rakibin HİÇ GÖRMEDİĞİ (en yeni) oyunlarındaki pozisyonlarda rakiple aynı
hamleyi oynuyor mu? Ölçüme şunlar GİRMEZ, çünkü skoru şişirirler:
  - açılış kitabındaki pozisyonlar (bot orada ezberden oynar),
  - tek yasal hamlesi olan pozisyonlar (tahmin edilecek bir şey yok).

TANIMLAR
========
  m  = botun ilk tahmininin rakibin hamlesiyle eşleşme oranı        (0-1)
  b  = aynı pozisyonlarda kişiselleştirilmemiş genel modelin oranı  (taban)
  s  = Stockfish'in en iyi hamlesinin eşleşme oranı                 (referans)
  T  = ulaşılabilir tavan. İnsanlar aynı pozisyonda bile her zaman aynı hamleyi
       oynamaz; kusursuz bir taklit bile 1.0'a ulaşamaz. T = max(0.75, rakibin
       tekrar eden pozisyonlardaki kendi kendisiyle tutarlılığı). 0.75 bir
       varsayımdır; veriden gelen tahmin onu yalnızca yükseltebilir, yani skoru
       yalnızca sıkılaştırabilir.

ALT SKORLAR (0-100)
===================
  Hamle uyumu  = 100 * m
  Açılış uyumu = 100 * (test oyunlarının ilk 8 hamlesinde rakibin seçimlerinin
                 kitapta bir seçenek olarak bulunma oranı)
  Hata profili = 100 * (0.5 * A + 0.5 * B)
       A = hata SIKLIĞI benzerliği: oyun aşamasına göre, botun hata oranı ile
           rakibin hata oranı ne kadar yakın (1 = aynı).
       B = hata YERİ benzerliği: rakibin hata yaptığı pozisyonlarda botun da
           hata yapma oranı.
       "Hata" = Stockfish'e göre kazanma şansını en az 10 puan düşüren hamle.
       Botun hamlesi burada ilk tahmini değil, gerçekte oynayacağı gibi
       olasılık dağılımından ÖRNEKLENEN hamledir.
  Güven        = eğitimdeki oyun sayısı ve ölçümün belirsizliği

ANA SKOR
========
  Ana = 100 * ( 0.6 * min(1, m / T) + 0.2 * Açılış + 0.2 * Hata )

  Hata profili hesaplanamadıysa (Stockfish yoksa) ağırlıklar 0.75 / 0.25 olur.
  Aynı formül genel model için de hesaplanır (aynı açılış kitabıyla) ve yan
  yana gösterilir. Bot, genel modelden istatistiksel olarak anlamlı biçimde iyi
  değilse (m - b farkının %95 güven aralığı sıfırı içeriyorsa) skor "belirsiz"
  olarak işaretlenir ve bu açıkça yazılır.

  Skorlar hep aşağı yuvarlanır; hiçbir skor olduğundan iyi gösterilmez.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import chess
import numpy as np

from .config import FULL_FINETUNE_GAMES, LOW_DATA_GAMES, TEMPO_LABELS_TR
from .encoding import position_key
from .games import GameRecord

DEFAULT_CEILING = 0.75
W_MOVE, W_OPENING, W_ERROR = 0.6, 0.2, 0.2
ERROR_THRESHOLD = 10.0     # kazanma şansı puanı
MIN_MEASURED_POSITIONS = 1000
PHASES = ("opening", "middlegame", "endgame")
PHASE_LABELS_TR = {"opening": "açılış", "middlegame": "orta oyun", "endgame": "oyun sonu"}

EXPECTATION_TR = (
    "İnsanlar aynı pozisyonda bile her zaman aynı hamleyi oynamaz. Bu yüzden hiçbir bot "
    "bir oyuncuyla %100 aynı oynayamaz; en iyi kişisel modeller bile 10 hamlenin 6-7'sinden "
    "fazlasını tutturamaz."
)


@dataclass
class Predictions:
    top1: np.ndarray       # (N,) modelin en olası gördüğü hamlenin indeksi
    p_target: np.ndarray   # (N,) rakibin oynadığı hamleye verilen olasılık
    p_top1: np.ndarray     # (N,) ilk tahmine verilen olasılık
    sample: np.ndarray | None = None  # (N,) dağılımdan örneklenen hamle (botun gerçekte oynayacağı)
    samples_by_t: dict | None = None  # sıcaklık -> (N,) örneklenen hamleler (kalibrasyon için)


# --------------------------------------------------------------------------
# Yardımcılar
# --------------------------------------------------------------------------

def win_percent(cp: np.ndarray | float) -> np.ndarray | float:
    """Santipiyon değerlendirmesini kazanma şansına (0-100) çevirir (Lichess formülü)."""
    cp = np.clip(np.asarray(cp, dtype=np.float64), -2000, 2000)
    return 50.0 + 50.0 * (2.0 / (1.0 + np.exp(-0.00368208 * cp)) - 1.0)


def floor_score(value: float) -> int:
    """0-100 aralığına sıkıştırıp AŞAĞI yuvarlar (skoru asla iyi gösterme)."""
    if value is None or math.isnan(value):
        return 0
    return int(math.floor(min(100.0, max(0.0, value)) + 1e-9))


def phases_from_codes(current_codes: np.ndarray, ply: np.ndarray) -> np.ndarray:
    """Her pozisyonun oyun aşaması. (N, 64) taş kodları ve yarım hamle sayısından.

    - oyun sonu: iki tarafın piyon ve şah dışındaki taş değeri toplamı <= 13
      (ör. birer kale + birer hafif taş ya da daha azı)
    - açılış: ilk 12 hamle (oyun sonu değilse)
    - orta oyun: gerisi
    """
    values = np.zeros(13, dtype=np.int32)
    for own, opp, v in ((2, 8, 3), (3, 9, 3), (4, 10, 5), (5, 11, 9)):
        values[own] = values[opp] = v
    material = values[current_codes.astype(np.int64)].sum(axis=1)
    out = np.full(current_codes.shape[0], "middlegame", dtype=object)
    out[ply < 24] = "opening"
    out[material <= 13] = "endgame"
    return out


def cluster_bootstrap(
    values: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Ortalamanın %95 güven aralığı; oyunlar (gruplar) yeniden örneklenir.

    Aynı oyunun hamleleri birbirinden bağımsız değildir; hamleleri tek tek
    örneklemek aralığı olduğundan dar gösterirdi.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"))
    _, inverse = np.unique(groups, return_inverse=True)
    n_groups = int(inverse.max()) + 1
    sums = np.bincount(inverse, weights=values, minlength=n_groups)
    counts = np.bincount(inverse, minlength=n_groups).astype(np.float64)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_groups, size=(n_boot, n_groups))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def estimate_ceiling(
    fit_games: Iterable[GameRecord], min_visits: int = 5, min_total: int = 300, book=None
) -> dict:
    """Rakibin kendi kendisiyle tutarlılığı: aynı pozisyona tekrar geldiğinde,
    diğer gelişlerinde en sık oynadığı hamleyi yine oynama oranı.

    `book` verilirse açılış kitabındaki pozisyonlar sayılmaz. Hamle uyumu kitap
    dışında ölçüldüğü için tavan da kitap dışında tahmin edilmeli: kitaptaki
    ezber hatlarda tutarlılık çok yüksektir (ilk gerçek veri denemesinde 0.91)
    ve kitap dışındaki zorluğu temsil etmez. Kitap dışında yeterince tekrar eden
    pozisyon yoksa varsayılan 0.75 kullanılır.

    Tavan olarak max(0.75, bu değer) kullanılır: veri tavanı yalnızca yükseltebilir.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for game in fit_games:
        board = chess.Board()
        player_turn = chess.WHITE if game.player_color == "white" else chess.BLACK
        for ply, uci in enumerate(game.moves[:60]):
            if board.turn == player_turn and (book is None or book.lookup(board, ply) is None):
                counts[position_key(board)][uci] += 1
            board.push(chess.Move.from_uci(uci))
    hits = visits = 0.0
    for moves in counts.values():
        n = sum(moves.values())
        if n < min_visits or len(moves) == 0:
            continue
        # Her pozisyon en çok 20 geliş kadar sayılır; başlangıç pozisyonu sonucu ezmesin.
        weight = min(n, 20) / n
        for uci, c in moves.items():
            others = dict(moves)
            others[uci] = c - 1
            top = max(others.values())
            tied = [u for u, v in others.items() if v == top]
            hit = (1.0 / len(tied)) if uci in tied else 0.0
            hits += weight * c * hit
            visits += weight * c
    estimate = hits / visits if visits >= min_total else None
    value = max(DEFAULT_CEILING, estimate) if estimate is not None else DEFAULT_CEILING
    return {
        "value": float(value),
        "self_consistency": None if estimate is None else float(estimate),
        "weighted_visits": float(visits),
        "source": "varsayım (0.75)" if value == DEFAULT_CEILING else "rakibin kendi kendisiyle tutarlılığı",
    }


# --------------------------------------------------------------------------
# Hata profili
# --------------------------------------------------------------------------

def error_profile(
    loss_player: np.ndarray,
    loss_bot: np.ndarray,
    phases: np.ndarray,
    threshold: float = ERROR_THRESHOLD,
) -> dict:
    """Kazanma şansı kayıplarından (0-100) hata profili benzerliği.

    Girdiler aynı pozisyonlara aittir: rakibin oynadığı hamlenin kaybı ve botun
    (örneklenen) hamlesinin kaybı.
    """
    loss_player = np.asarray(loss_player, dtype=np.float64)
    loss_bot = np.asarray(loss_bot, dtype=np.float64)
    ok = ~(np.isnan(loss_player) | np.isnan(loss_bot))
    loss_player, loss_bot, phases = loss_player[ok], loss_bot[ok], np.asarray(phases)[ok]
    n = int(ok.sum())
    if n == 0:
        return {"available": False, "n": 0}
    err_p = loss_player >= threshold
    err_b = loss_bot >= threshold

    per_phase, a_sum = {}, 0.0
    for phase in PHASES:
        sel = phases == phase
        k = int(sel.sum())
        if k == 0:
            per_phase[phase] = {"n": 0, "player_rate": None, "bot_rate": None}
            continue
        rp, rb = float(err_p[sel].mean()), float(err_b[sel].mean())
        closeness = 1.0 if max(rp, rb) == 0 else 1.0 - abs(rp - rb) / max(rp, rb)
        a_sum += (k / n) * closeness
        per_phase[phase] = {"n": k, "player_rate": rp, "bot_rate": rb}

    n_player_errors = int(err_p.sum())
    b = float(err_b[err_p].mean()) if n_player_errors else 0.0
    return {
        "available": True,
        "n": n,
        "rate_similarity": float(a_sum),               # A
        "co_occurrence": b,                            # B
        "score": 100.0 * (0.5 * a_sum + 0.5 * b),
        "player_error_rate": float(err_p.mean()),
        "bot_error_rate": float(err_b.mean()),
        "n_player_errors": n_player_errors,
        "few_errors": n_player_errors < 30,
        "per_phase": per_phase,
        "threshold": threshold,
    }


# --------------------------------------------------------------------------
# Güven
# --------------------------------------------------------------------------

CONFIDENCE = {
    "low": {"label": "Düşük", "bar": 20},
    "medium": {"label": "Orta", "bar": 40},
    "high": {"label": "Yüksek", "bar": 80},
}
_ORDER = ["low", "medium", "high"]


def confidence_level(n_games: int, n_measured: int, ci_halfwidth: float) -> tuple[str, list[str]]:
    """Güven seviyesi ve düşürülme nedenleri."""
    reasons: list[str] = []
    if n_games < LOW_DATA_GAMES:
        level = "low"
        reasons.append(f"yalnızca {n_games} oyun")
    elif n_games < FULL_FINETUNE_GAMES:
        level = "medium"
    else:
        level = "high"
    uncertain_measure = n_measured < MIN_MEASURED_POSITIONS or (not math.isnan(ci_halfwidth) and ci_halfwidth > 0.03)
    if uncertain_measure and level != "low":
        level = _ORDER[_ORDER.index(level) - 1]
        reasons.append(f"ölçüm yalnızca {n_measured} pozisyona dayanıyor")
    return level, reasons


# --------------------------------------------------------------------------
# Ana hesap
# --------------------------------------------------------------------------

def main_score(m: float, ceiling: float, opening: float, error_score: float | None) -> float:
    """Ana skor (0-100). `opening` 0-1, `error_score` 0-100 ya da None."""
    move_part = min(1.0, m / ceiling) if ceiling > 0 else 0.0
    if error_score is None:
        w_move, w_open = W_MOVE / (W_MOVE + W_OPENING), W_OPENING / (W_MOVE + W_OPENING)
        return 100.0 * (w_move * move_part + w_open * opening)
    return 100.0 * (W_MOVE * move_part + W_OPENING * opening + W_ERROR * error_score / 100.0)


def compute_scores(
    *,
    player: str,
    tempo: str,
    n_games: int,
    target: np.ndarray,          # (N,) test örneklerinde rakibin hamlesi
    measured: np.ndarray,        # (N,) bool: kitap dışı ve zorunlu olmayan pozisyonlar
    groups: np.ndarray,          # (N,) oyun indeksi
    bot: Predictions,
    generic: Predictions,
    opening: dict,               # book.opening_coverage çıktısı
    ceiling: dict,               # estimate_ceiling çıktısı
    stockfish_best: np.ndarray | None = None,   # (N,) Stockfish'in en iyi hamlesi, bilinmiyorsa -1
    error_bot: dict | None = None,              # error_profile çıktısı (bot)
    error_generic: dict | None = None,          # error_profile çıktısı (genel model)
    n_test_games: int = 0,
    excluded: dict | None = None,
    seed: int = 0,
) -> dict:
    target = np.asarray(target)
    rows = np.asarray(measured, dtype=bool)
    n_measured = int(rows.sum())
    label = TEMPO_LABELS_TR.get(tempo, tempo)
    notes: list[str] = []

    hit_bot = (bot.top1[rows] == target[rows]).astype(np.float64)
    hit_gen = (generic.top1[rows] == target[rows]).astype(np.float64)
    g = np.asarray(groups)[rows]
    m = float(hit_bot.mean()) if n_measured else 0.0
    b = float(hit_gen.mean()) if n_measured else 0.0
    m_ci = cluster_bootstrap(hit_bot, g, seed=seed)
    diff_ci = cluster_bootstrap(hit_bot - hit_gen, g, seed=seed + 1)
    significant_generic = bool(n_measured and diff_ci[0] > 0)

    sf = None
    significant_sf = None
    sf_ci = None
    if stockfish_best is not None:
        known = rows & (np.asarray(stockfish_best) >= 0)
        if known.sum():
            hit_sf = (np.asarray(stockfish_best)[known] == target[known]).astype(np.float64)
            hit_bot_k = (bot.top1[known] == target[known]).astype(np.float64)
            sf = float(hit_sf.mean())
            sf_ci = cluster_bootstrap(hit_bot_k - hit_sf, np.asarray(groups)[known], seed=seed + 2)
            significant_sf = bool(sf_ci[0] > 0)

    t = float(ceiling["value"])
    coverage = float(opening.get("coverage", 0.0))
    err_available = bool(error_bot and error_bot.get("available"))
    err_score = float(error_bot["score"]) if err_available else None
    err_score_gen = float(error_generic["score"]) if (error_generic and error_generic.get("available")) else None
    if not err_available:
        notes.append("Hata profili hesaplanamadı (Stockfish değerlendirmesi yok); ana skor yalnızca hamle ve açılış uyumundan oluşuyor.")

    main = main_score(m, t, coverage, err_score)
    main_generic = main_score(b, t, coverage, err_score_gen if err_available else None)

    half = (m_ci[1] - m_ci[0]) / 2 if n_measured else float("nan")
    level, reasons = confidence_level(n_games, n_measured, half)

    # --- karar cümlesi ----------------------------------------------------
    if n_measured == 0:
        verdict = "Ölçülebilecek pozisyon kalmadı; skor hesaplanamadı."
        state = "uncertain"
    elif not significant_generic:
        verdict = (
            f"Bu bot, {player} için özel eğitilmemiş genel bir modelden anlamlı şekilde daha iyi değil "
            f"(bot %{100 * m:.0f}, genel model %{100 * b:.0f}). Kişiselleştirme bu veriyle işe yaramamış olabilir."
        )
        state = "uncertain"
    else:
        verdict = (
            f"Bot, genel modelden anlamlı şekilde daha iyi: aynı pozisyonlarda {player} ile "
            f"%{100 * m:.0f} eşleşiyor, genel model %{100 * b:.0f}."
        )
        state = "ok"
    if sf is not None and significant_sf is False:
        verdict += f" Ayrıca Stockfish'in en iyi hamlesinden (%{100 * sf:.0f}) anlamlı şekilde daha iyi değil."
        state = "uncertain"
    if level == "low":
        state = "uncertain"

    ten = int(round(10 * m))
    sentence = (
        f"Bu bot, {player} adlı oyuncunun hiç görmediği oyunlarında her 10 hamlenin yaklaşık {ten}'"
        f"{_suffix_inde(ten)} onunla aynı hamleyi oynadı."
    )

    conf_text = f"{n_games} {label} oyun ile eğitildi"
    if reasons:
        conf_text += " (" + ", ".join(reasons) + ")"

    if err_available:
        err_text = (
            f"{player} hata yaptığında bot da %{100 * error_bot['co_occurrence']:.0f} ihtimalle hata yapıyor; "
            f"hata sıklıkları %{100 * error_bot['rate_similarity']:.0f} oranında benziyor."
        )
        if error_bot.get("few_errors"):
            err_text += " Az sayıda hata incelendi, bu skor kaba bir tahmindir."
    else:
        err_text = "Hesaplanamadı."

    return {
        "version": 1,
        "player": player,
        "tempo": tempo,
        "main": {
            "score": floor_score(main),
            "state": state,                      # "ok" | "uncertain" (gri göster)
            "sentence": sentence,
            "generic_score": floor_score(main_generic),
            "verdict": verdict,
        },
        "sub": {
            "opening": {
                "score": floor_score(100 * coverage),
                "text": f"{player} adlı oyuncunun yeni oyunlarındaki açılış hamlelerinin %{floor_score(100 * coverage)} kadarı botun repertuvarında var.",
            },
            "move": {
                "score": floor_score(100 * m),
                "text": f"Açılıştan sonraki her 10 hamlenin yaklaşık {ten}'{_suffix_i(ten)} birebir aynı.",
                "ci": [floor_score(100 * m_ci[0]), floor_score(100 * m_ci[1])] if n_measured else None,
            },
            "error": {
                "score": floor_score(err_score) if err_available else None,
                "available": err_available,
                "text": err_text,
            },
            "confidence": {
                "level": level,
                "label": CONFIDENCE[level]["label"],
                "bar": CONFIDENCE[level]["bar"],
                "text": conf_text,
            },
        },
        "references": {
            "bot": m,
            "generic": b,
            "stockfish": sf,
            "bot_ci": list(m_ci),
            "bot_minus_generic_ci": list(diff_ci),
            "bot_minus_stockfish_ci": None if sf_ci is None else list(sf_ci),
            "significant_vs_generic": significant_generic,
            "significant_vs_stockfish": significant_sf,
        },
        "formula": {
            "text": "Ana = 100 x (0.6 x min(1, m/T) + 0.2 x Açılış + 0.2 x Hata)",
            "m": m, "b": b, "T": t,
            "ceiling": ceiling,
            "weights": {"move": W_MOVE, "opening": W_OPENING, "error": W_ERROR},
            "opening": coverage,
            "error": err_score,
            "error_detail": error_bot if err_available else None,
            "error_detail_generic": error_generic if (error_generic and error_generic.get("available")) else None,
            "raw_main": main,
            "raw_main_generic": main_generic,
            "error_threshold": ERROR_THRESHOLD,
        },
        "counts": {
            "train_games": n_games,
            "test_games": n_test_games,
            "test_positions": int(target.shape[0]),
            "measured_positions": n_measured,
            "excluded": excluded or {},
        },
        "notes": notes,
        "expectation": EXPECTATION_TR,
    }


def _suffix_inde(n: int) -> str:
    """6'sında, 5'inde, 9'unda ... (Türkçe ünlü uyumu)."""
    return {0: "ında", 1: "inde", 2: "sinde", 3: "ünde", 4: "ünde", 5: "inde", 6: "sında", 7: "sinde", 8: "inde", 9: "unda", 10: "unda"}[n]


def _suffix_i(n: int) -> str:
    """6'sı, 5'i, 9'u ..."""
    return {0: "ı", 1: "i", 2: "si", 3: "ü", 4: "ü", 5: "i", 6: "sı", 7: "si", 8: "i", 9: "u", 10: "u"}[n]


def format_scores_text(scores: dict) -> str:
    """Skorların düz metin özeti (not defteri çıktısı için)."""

    def bar(value: int | None) -> str:
        filled = 0 if value is None else max(0, min(10, value // 10))
        return "█" * filled + "░" * (10 - filled)

    main, sub = scores["main"], scores["sub"]
    uncertain = main["state"] != "ok"
    lines = [
        f"RAKİP BENZERLİK SKORU: {main['score']}/100" + ("  (BELİRSİZ)" if uncertain else ""),
        f"\"{main['sentence']}\"",
        "",
        f"├─ Açılış uyumu    {bar(sub['opening']['score'])}  {sub['opening']['score']:>3}  ({sub['opening']['text']})",
        f"├─ Hamle uyumu     {bar(sub['move']['score'])}  {sub['move']['score']:>3}  ({sub['move']['text']})",
    ]
    if sub["error"]["available"]:
        lines.append(f"├─ Hata profili    {bar(sub['error']['score'])}  {sub['error']['score']:>3}  ({sub['error']['text']})")
    else:
        lines.append(f"├─ Hata profili    {bar(None)}    -  ({sub['error']['text']})")
    conf = sub["confidence"]
    lines.append(f"└─ Güven seviyesi  {bar(conf['bar'])}  {conf['label']}  ({conf['text']})")
    refs = scores["references"]
    lines += [
        "",
        f"Karşılaştırma (aynı pozisyonlarda rakiple aynı hamle): bot %{100 * refs['bot']:.1f}, "
        f"genel model %{100 * refs['generic']:.1f}"
        + (f", Stockfish %{100 * refs['stockfish']:.1f}" if refs.get("stockfish") is not None else ""),
        f"Genel modelin aynı formülle skoru: {main['generic_score']}/100",
        main["verdict"],
    ]
    lines += [f"Not: {n}" for n in scores.get("notes", [])]
    lines += ["", scores["expectation"]]
    return "\n".join(lines)
