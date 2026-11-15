"""Kullanıcının yüklediği PGN dosyaları (Kaggle veri seti ya da Colab yüklemesi)."""

from __future__ import annotations

import zipfile
from collections import Counter
from pathlib import Path

from ..games import GameRecord, PlayerMatcher, parse_pgn_text


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def load_paths(
    paths: list[str | Path],
    matcher: PlayerMatcher,
    stats: Counter | None = None,
) -> list[GameRecord]:
    """Verilen dosya ve klasörlerdeki tüm .pgn / .zip dosyalarını okur."""
    stats = stats if stats is not None else Counter()
    files: list[Path] = []
    for entry in paths:
        entry = Path(entry)
        if entry.is_dir():
            files.extend(sorted(p for p in entry.rglob("*") if p.suffix.lower() in (".pgn", ".zip")))
        elif entry.exists():
            files.append(entry)
        else:
            stats["missing_pgn_paths"] += 1
    records: list[GameRecord] = []
    for file in files:
        if file.suffix.lower() == ".zip":
            with zipfile.ZipFile(file) as archive:
                for name in archive.namelist():
                    if name.lower().endswith(".pgn"):
                        raw = archive.read(name)
                        try:
                            text = raw.decode("utf-8")
                        except UnicodeDecodeError:
                            text = raw.decode("latin-1")
                        records.extend(parse_pgn_text(text, matcher, "pgn", stats=stats))
        else:
            records.extend(parse_pgn_text(_read_text(file), matcher, "pgn", stats=stats))
    return records
