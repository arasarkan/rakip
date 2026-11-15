"""Chess.com public API: aylık arşivlerden oyun indirir.

Belge: https://www.chess.com/news/view/published-data-api
Arşivler aylıktır ve tüm tempoları içerir. İstekler sırayla yapılır (paralel
istek 429 üretir). Geçmiş aylar değişmez, kalıcı önbelleğe alınır; içinde
bulunulan ay bir saatlik önbellekle tutulur.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter

from ..games import GameRecord, PlayerMatcher, parse_pgn_text
from ..http import NotFound, RateLimitedFetcher

API = "https://api.chess.com/pub"
TIME_CLASSES = {"rapid": "rapid", "blitz": "blitz", "bullet": "bullet", "daily": "correspondence"}


def fetch_games(
    username: str,
    fetcher: RateLimitedFetcher,
    max_months: int | None = None,
    stats: Counter | None = None,
    today: dt.date | None = None,
) -> list[GameRecord]:
    """Kullanıcının tüm tempolardaki dereceli standart oyunlarını indirir.

    Chess.com'da "klasik" sınıfı yoktur; 25 dakika ve üstü oyunlar burada
    süre hesabıyla klasik sayılır, gerisi sitenin kendi sınıfını kullanır.
    """
    stats = stats if stats is not None else Counter()
    user = username.strip().lower()
    today = today or dt.datetime.now(dt.timezone.utc).date()
    current_month = f"{today.year}/{today.month:02d}"

    index_path = fetcher.fetch(f"{API}/player/{user}/games/archives", ttl=3600, suffix=".json")
    archives = json.loads(index_path.read_text(encoding="utf-8")).get("archives", [])
    archives = sorted(archives, reverse=True)  # en yeni ay önce
    if max_months:
        archives = archives[:max_months]

    matcher = PlayerMatcher([username], exact=True)
    records: list[GameRecord] = []
    for url in archives:
        ttl = 3600 if url.endswith(current_month) else None
        try:
            path = fetcher.fetch(url, ttl=ttl, suffix=".json")
        except NotFound:
            continue
        for game in json.loads(path.read_text(encoding="utf-8")).get("games", []):
            stats["seen"] += 1
            if game.get("rules") != "chess":
                stats["dropped_variant_or_custom_start"] += 1
                continue
            if not game.get("rated", False):
                stats["dropped_unrated"] += 1
                continue
            pgn = game.get("pgn")
            if not pgn:
                stats["dropped_no_pgn"] += 1
                continue
            site_class = TIME_CLASSES.get(game.get("time_class", ""))
            if site_class == "correspondence":
                stats["dropped_correspondence"] += 1
                continue
            # tempo_hint verilmez: süre hesabı klasik oyunları da ayırabilsin.
            parsed = parse_pgn_text(pgn, matcher, "chesscom", stats=Counter(), prefilter=False)
            if not parsed:
                stats["dropped_unparsed"] += 1
                continue
            record = parsed[0]
            if record.tempo != "classical" and site_class:
                record.tempo = site_class
            record.site = game.get("url") or record.site
            stats["kept"] += 1
            records.append(record)
    return records


def tempo_counts(records: list[GameRecord]) -> dict[str, int]:
    return dict(Counter(r.tempo for r in records))
