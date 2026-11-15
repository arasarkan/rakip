"""Lichess API: bir kullanıcının oyunlarını PGN olarak indirir.

Belge: https://lichess.org/api#tag/Games/operation/apiGamesUser
Oyun dışa aktarımı tek, akan bir istektir (anonim ~20 oyun/sn). Lichess aynı
anda tek istek yapılmasını ve 429 alınırsa bir dakika beklenmesini ister;
ikisini de `RateLimitedFetcher` sağlar.
"""

from __future__ import annotations

import json
from collections import Counter

from ..games import GameRecord, PlayerMatcher, parse_pgn_text
from ..http import RateLimitedFetcher

API = "https://lichess.org/api"
PERF_TYPES = {"classical": "classical", "rapid": "rapid", "blitz": "blitz", "bullet": "bullet"}


def _auth(token: str | None) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def tempo_counts(username: str, fetcher: RateLimitedFetcher, token: str | None = None) -> dict[str, int]:
    """Kullanıcının her tempodaki dereceli oyun sayısı (oyunları indirmeden)."""
    path = fetcher.fetch(
        f"{API}/user/{username}", headers={"Accept": "application/json", **_auth(token)}, ttl=3600, suffix=".json"
    )
    perfs = json.loads(path.read_text(encoding="utf-8")).get("perfs", {})
    return {tempo: int(perfs.get(perf, {}).get("games", 0)) for tempo, perf in PERF_TYPES.items()}


def fetch_games(
    username: str,
    tempo: str,
    fetcher: RateLimitedFetcher,
    max_games: int | None = 20000,
    token: str | None = None,
    stats: Counter | None = None,
    ttl: float | None = 6 * 3600,
) -> list[GameRecord]:
    """Seçilen tempodaki dereceli standart oyunları (en yeniden eskiye) indirir."""
    params = {
        "perfType": PERF_TYPES[tempo],
        "rated": "true",
        "clocks": "true",
        "opening": "true",
        "evals": "false",
    }
    if max_games:
        params["max"] = str(max_games)
    path = fetcher.fetch(
        f"{API}/games/user/{username}",
        params=params,
        headers={"Accept": "application/x-chess-pgn", **_auth(token)},
        ttl=ttl,
        suffix=".pgn",
    )
    text = path.read_text(encoding="utf-8", errors="replace")
    matcher = PlayerMatcher([username], exact=True)
    return parse_pgn_text(text, matcher, "lichess", tempo_hint=tempo, stats=stats, prefilter=False)
