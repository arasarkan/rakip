"""Proje genelindeki sabitler ve tempo sınıflandırması."""

from __future__ import annotations

import re

# Ürünün desteklediği tempolar. "bullet" veri olarak tutulur ama hazırlık
# temposu olarak önerilmez.
TEMPOS = ("classical", "rapid", "blitz", "bullet")

TEMPO_LABELS_TR = {
    "classical": "klasik",
    "rapid": "rapid",
    "blitz": "blitz",
    "bullet": "bullet",
}

# Veri miktarı eşikleri (seçilen tempodaki oyun sayısı).
LOW_DATA_GAMES = 200      # altı: güven "Düşük", arayüzde uyarı
FULL_FINETUNE_GAMES = 1000  # altı: dondurulmuş gövde + oyuncu embedding'i

# Temizlik: bundan kısa oyunlar (yarım hamle) atılır.
MIN_PLIES = 10

# Temel model.
BASE_MODEL_REPO = "UofTCSSLab/Maia3-5M"
BASE_MODEL_FILE = "maia3-5m.pt"

# Hub'da eğitilmiş rakip depolarını bulmak için kullanılan etiket.
HUB_TAG = "rakip-bot"

USER_AGENT = "rakip-chess-trainer/0.1"

_TC_SIMPLE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:\+\s*(\d+(?:\.\d+)?))?\s*$")


def parse_time_control(tc: str | None) -> tuple[float, float] | None:
    """"300+3" -> (300.0, 3.0). Tanınmayan biçimlerde None döner."""
    if not tc:
        return None
    m = _TC_SIMPLE.match(tc)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2) or 0.0)


def classify_time_control(tc: str | None) -> str | None:
    """PGN TimeControl alanından tempo çıkarır (Lichess eşikleriyle).

    Tahmini süre = ana süre + 40 x eklenti. Yazışmalı oyunlar için
    "correspondence", belirlenemeyenler için None döner.
    """
    if tc is None:
        return None
    tc = tc.strip()
    if tc in ("", "?"):
        return None
    if tc == "-" or "/" in tc:
        # "-" Lichess yazışmalı; "1/86400" Chess.com günlük oyun.
        # "40/7200:3600" gibi OTB biçimleri de buraya düşer; onları çağıran
        # taraf etkinlik adından sınıflandırır.
        return "correspondence" if tc == "-" or tc.startswith("1/") else None
    parsed = parse_time_control(tc)
    if parsed is None:
        return None
    base, inc = parsed
    estimate = base + 40 * inc
    if estimate < 180:
        return "bullet"
    if estimate < 480:
        return "blitz"
    if estimate < 1500:
        return "rapid"
    return "classical"


# Adında "rapid" geçmeyen ama hızlı tempoda oynanan bilinen etkinlik serileri.
# TWIC bu etkinliklerde saat bilgisi vermez; adına bakılmazsa klasik sanılırlar.
_RAPID_SERIES = (
    "global chess league", "champions chess tour", "chessable masters", "meltwater",
    "magnus carlsen invitational", "carlsen invitational", "airthings masters", "aimchess",
    "ftx crypto cup", "crypto cup", "oslo esports", "generation cup", "new in chess classic",
    "skilling open", "opera euro", "goldmoney asian", "lindores abbey", "chess.com classic",
    "charity cup", "american cup", "tata steel india", "superbet poland", "superunited",
    "croatia grand chess tour", "esports world cup",
)
# Blitz ve bullet etkinlikleri. TWIC haftalık Titled Tuesday turnuvalarını
# "Titled Tue 2nd Jan Early" gibi kısaltarak yazar; bu yüzden "titled tue" aranır.
_BLITZ_WORDS = (
    "blitz", "bullet", "armageddon", "titled tue", "titled arena", "speed chess",
    "bullet brawl", "lichess titled", "hourly",
)
# Bu "Site" değerleri oyunun çevrim içi oynandığını gösterir.
_ONLINE_SITES = ("chess.com", "lichess", "chess24", "internet", "online", "playchess", "icc")


def is_online_site(site: str | None) -> bool:
    text = (site or "").casefold()
    return any(word in text for word in _ONLINE_SITES)


def classify_otb_event(event: str | None, site: str | None = None) -> str | None:
    """Saat bilgisi olmayan oyunlarda tempoyu etkinlik adından çıkarır.

    Sıra: açık kelimeler (blitz, armageddon, titled tue ...; sonra rapid), sonra
    adında tempo geçmeyen bilinen hızlı seriler. Hiçbiri yoksa:
    - masa başı oyun -> klasik (masa başı turnuvaların çoğu klasiktir),
    - çevrim içi oyun (Site: Chess.com, Lichess ...) -> None, yani "bilinmiyor".
      Çevrim içi klasik oyun nadirdir; tahmin etmek yerine oyun atılır ve
      atılan sayısı raporda görünür.
    Karma etkinliklerde TWIC rapid ve blitz bölümlerini genellikle ayrı adlarla
    yazar; açık kelime her zaman seri adından önce gelir.
    """
    text = " ".join((event or "").casefold().split())
    if any(word in text for word in _BLITZ_WORDS):
        return "blitz"
    if "rapid" in text or "rapide" in text or "schnell" in text:
        return "rapid"
    if any(name in text for name in _RAPID_SERIES):
        return "rapid"
    if is_online_site(site):
        return None
    return "classical"


# Ek eğitim verisi için varsayılan komşu tempolar (bullet, blitz'e yalnızca blitz hedefinde eklenir).
NEIGHBOUR_TEMPOS = {
    "classical": ["rapid"],
    "rapid": ["classical", "blitz"],
    "blitz": ["rapid", "bullet"],
    "bullet": ["blitz"],
}
