"""The Week in Chess (TWIC): masa başı (OTB) oyunların haftalık PGN arşivleri.

Her sayı ayrı bir zip dosyasıdır: https://theweekinchess.com/zips/twic<N>g.zip
Sayılar değişmez, kalıcı önbelleğe alınır. Dosyalar yalnızca Kaggle/Colab
tarafında indirilir.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections import Counter

from ..games import GameRecord, PlayerMatcher, parse_pgn_text
from ..http import NotFound, RateLimitedFetcher

BASE = "https://theweekinchess.com"
FIRST_ZIP_ISSUE = 920  # zip arşivi bu sayıdan itibaren mevcut


def latest_issue(fetcher: RateLimitedFetcher) -> int:
    path = fetcher.fetch(f"{BASE}/twic", ttl=24 * 3600, suffix=".html")
    numbers = [int(n) for n in re.findall(r"twic(\d{3,5})g\.zip", path.read_text(encoding="utf-8", errors="replace"))]
    if not numbers:
        raise RuntimeError("TWIC sayfasında sayı numarası bulunamadı; twic_to parametresini elle verin")
    return max(numbers)


def read_issue(fetcher: RateLimitedFetcher, issue: int) -> str | None:
    try:
        path = fetcher.fetch(f"{BASE}/zips/twic{issue}g.zip", suffix=".zip")
    except NotFound:
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            parts = [
                archive.read(name).decode("latin-1")
                for name in archive.namelist()
                if name.lower().endswith(".pgn")
            ]
    except zipfile.BadZipFile:
        return None
    return "\n\n".join(parts)


def fetch_games(
    matcher: PlayerMatcher,
    fetcher: RateLimitedFetcher,
    issues: int = 150,
    issue_to: int | None = None,
    issue_from: int | None = None,
    stats: Counter | None = None,
    log=print,
) -> list[GameRecord]:
    """Son `issues` sayıda (ya da verilen aralıkta) oyuncunun oyunlarını toplar."""
    stats = stats if stats is not None else Counter()
    issue_to = issue_to or latest_issue(fetcher)
    issue_from = max(FIRST_ZIP_ISSUE, issue_from or (issue_to - issues + 1))
    records: list[GameRecord] = []
    for issue in range(issue_to, issue_from - 1, -1):
        text = read_issue(fetcher, issue)
        if text is None:
            stats["twic_missing_issues"] += 1
            continue
        found = parse_pgn_text(text, matcher, "twic", stats=stats)
        records.extend(found)
        if found:
            log(f"[twic] sayı {issue}: {len(found)} oyun")
    return records


def parse_zip_bytes(data: bytes, matcher: PlayerMatcher, stats: Counter | None = None) -> list[GameRecord]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        text = "\n\n".join(
            archive.read(n).decode("latin-1") for n in archive.namelist() if n.lower().endswith(".pgn")
        )
    return parse_pgn_text(text, matcher, "twic", stats=stats)
