"""Oyun kaydı, PGN ayrıştırma, oyuncu eşleştirme ve temizlik.

Her kaynak (Lichess, Chess.com, TWIC, yüklenen PGN) oyunları aynı
`GameRecord` biçimine çevirir. Kayıtlar hedef oyuncunun bakış açısından
tutulur: `player_color`, `player_rating`, `result` hep hedef oyuncuya aittir.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import chess
import chess.pgn

from .config import MIN_PLIES, classify_otb_event, classify_time_control

STANDARD_VARIANTS = {"", "standard", "chess", "from position"}


@dataclass
class GameRecord:
    game_id: str
    source: str                  # lichess | chesscom | twic | pgn
    player_color: str            # white | black
    tempo: str                   # classical | rapid | blitz | bullet
    date: str | None             # ISO: YYYY-MM-DD ya da YYYY-MM-DDTHH:MM:SS
    result: float | None         # hedef oyuncu için 1 / 0.5 / 0
    player_rating: int | None
    opponent_rating: int | None
    opponent_name: str
    time_control: str | None
    moves: list[str]             # UCI, baştan sona iki tarafın hamleleri
    clocks: list[float | None] = field(default_factory=list)  # hamleden sonra kalan saniye
    eco: str | None = None
    opening: str | None = None
    termination: str | None = None
    site: str | None = None

    @property
    def sort_key(self) -> tuple[str, str]:
        # Tarihi bilinmeyen oyunlar "en eski" sayılır: en yeni olduklarını
        # kanıtlayamayız, bu yüzden test setine giremezler.
        return (self.date or "0000-00-00", self.game_id)

    @property
    def has_clocks(self) -> bool:
        return any(c is not None for c in self.clocks)

    def player_ply_indices(self) -> range:
        """Hedef oyuncunun hamle yaptığı yarım hamle indeksleri."""
        return range(0 if self.player_color == "white" else 1, len(self.moves), 2)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "GameRecord":
        return cls(**json.loads(line))


# --------------------------------------------------------------------------
# Oyuncu eşleştirme
# --------------------------------------------------------------------------

# NFKD ile ayrışmayan harfler (Türkçe noktasız ı dahil).
_TRANSLIT = str.maketrans({"ı": "i", "İ": "i", "ø": "o", "Ø": "o", "ł": "l", "Ł": "l", "đ": "d", "Đ": "d", "ß": "ss", "æ": "ae", "Æ": "ae"})


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.translate(_TRANSLIT))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9, ]+", " ", text.casefold()).strip()


def _split_name(name: str) -> tuple[str, str]:
    """"Carlsen, Magnus" / "Magnus Carlsen" / "Carlsen,M" -> (soyad, ad)."""
    norm = _normalize(name)
    if "," in norm:
        surname, _, given = norm.partition(",")
        return " ".join(surname.split()), " ".join(given.replace(",", " ").split())
    parts = norm.split()
    if not parts:
        return "", ""
    return parts[-1], " ".join(parts[:-1])


class PlayerMatcher:
    """PGN başlıklarından hedef oyuncunun rengini bulur.

    - `exact=True`: çevrim içi kullanıcı adları; büyük/küçük harf duyarsız tam eşleşme.
    - `exact=False`: masa başı isimler. Soyad eşit ve adlar uyumlu olmalı
      ("M" ile "Magnus" uyumludur, TWIC "Carlsen,M" yazar). İsimler çakışabildiği
      için `fide_id` verilmesi önerilir; başlıkta FIDE ID varsa o esas alınır.
    """

    def __init__(self, names: Iterable[str], fide_id: str | int | None = None, exact: bool = False):
        self.exact = exact
        self.fide_id = str(fide_id).strip() if fide_id else None
        names = [n for n in names if n and n.strip()]
        if not names and not self.fide_id:
            raise ValueError("En az bir oyuncu adı ya da FIDE ID gerekli")
        self._exact_names = {n.strip().casefold() for n in names}
        self._split_names = [_split_name(n) for n in names]
        # Toplu PGN'lerde hızlı ön eleme için aranan parçalar.
        self.needles = sorted({s for s, _ in self._split_names if s} | ({self.fide_id} if self.fide_id else set()))

    def _name_matches(self, header_name: str) -> bool:
        if self.exact:
            return header_name.strip().casefold() in self._exact_names
        surname, given = _split_name(header_name)
        for q_surname, q_given in self._split_names:
            if surname != q_surname:
                continue
            if not given or not q_given:
                return True
            a, b = given.split()[0], q_given.split()[0]
            if a.startswith(b) or b.startswith(a):
                return True
        return False

    def _side_matches(self, headers, side: str) -> bool:
        if self.fide_id:
            header_id = (headers.get(f"{side}FideId") or "").strip()
            if header_id and header_id not in ("0", "?"):
                return header_id == self.fide_id
        if not self._split_names:
            return False
        return self._name_matches(headers.get(side, ""))

    def color(self, headers) -> str | None:
        white = self._side_matches(headers, "White")
        black = self._side_matches(headers, "Black")
        if white == black:  # hiçbiri ya da (belirsiz) ikisi birden
            return None
        return "white" if white else "black"

    def maybe_in(self, text_casefolded: str) -> bool:
        """Ucuz ön eleme: ham PGN parçasında oyuncunun izi var mı?"""
        if self.exact:
            return any(n in text_casefolded for n in self._exact_names)
        return any(n in text_casefolded for n in self.needles)


# --------------------------------------------------------------------------
# PGN -> GameRecord
# --------------------------------------------------------------------------

def _parse_date(headers) -> str | None:
    raw = headers.get("UTCDate") or headers.get("Date") or ""
    m = re.match(r"^(\d{4})[.\-/](\d{2}|\?\?)[.\-/](\d{2}|\?\?)", raw.strip())
    if not m:
        return None
    year, month, day = m.group(1), m.group(2), m.group(3)
    month = "01" if "?" in month else month
    day = "01" if "?" in day else day
    date = f"{year}-{month}-{day}"
    time_raw = (headers.get("UTCTime") or "").strip()
    if headers.get("UTCDate") and re.match(r"^\d{2}:\d{2}:\d{2}$", time_raw):
        date += "T" + time_raw
    return date


def _parse_rating(value: str | None) -> int | None:
    try:
        rating = int((value or "").strip().rstrip("?"))
    except ValueError:
        return None
    return rating if 100 <= rating <= 4000 else None


_RESULT_WHITE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def _game_id(source: str, date: str | None, white: str, black: str, moves: list[str]) -> str:
    # Kaynak kimliğe girmez: aynı oyun iki kaynaktan gelirse aynı kimliği alır
    # ve tekilleştirmede elenir.
    key = "|".join([(date or "")[:10], white.casefold(), black.casefold(), " ".join(moves)])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def record_from_game(
    game: chess.pgn.Game,
    matcher: PlayerMatcher,
    source: str,
    tempo_hint: str | None = None,
    stats: Counter | None = None,
) -> GameRecord | None:
    """Tek bir PGN oyununu kayda çevirir; temizlik kurallarına takılırsa None döner.

    Atılma nedenleri `stats` sayacına yazılır.
    """
    stats = stats if stats is not None else Counter()
    headers = game.headers

    def drop(reason: str) -> None:
        stats[f"dropped_{reason}"] += 1
        return None

    color = matcher.color(headers)
    if color is None:
        return drop("player_not_found")

    variant = (headers.get("Variant") or "").strip().casefold()
    if variant not in STANDARD_VARIANTS or headers.get("FEN") or headers.get("SetUp") == "1":
        return drop("variant_or_custom_start")

    if game.errors:
        return drop("illegal_moves")

    moves: list[str] = []
    clocks: list[float | None] = []
    for node in game.mainline():
        moves.append(node.move.uci())
        clocks.append(node.clock())
    if len(moves) < MIN_PLIES:
        return drop("too_short")

    tc = (headers.get("TimeControl") or "").strip() or None
    tempo = tempo_hint or classify_time_control(tc)
    if tempo is None and source in ("twic", "pgn"):
        tempo = classify_otb_event(headers.get("Event"))
    if tempo == "correspondence":
        return drop("correspondence")
    if tempo is None:
        return drop("unknown_tempo")

    white_result = _RESULT_WHITE.get((headers.get("Result") or "").strip())
    result = None if white_result is None else (white_result if color == "white" else 1.0 - white_result)

    white, black = headers.get("White", "?"), headers.get("Black", "?")
    date = _parse_date(headers)
    me, opp = ("White", "Black") if color == "white" else ("Black", "White")
    stats["kept"] += 1
    return GameRecord(
        game_id=_game_id(source, date, white, black, moves),
        source=source,
        player_color=color,
        tempo=tempo,
        date=date,
        result=result,
        player_rating=_parse_rating(headers.get(f"{me}Elo")),
        opponent_rating=_parse_rating(headers.get(f"{opp}Elo")),
        opponent_name=headers.get(opp, "?"),
        time_control=tc,
        moves=moves,
        clocks=clocks if any(c is not None for c in clocks) else [],
        eco=headers.get("ECO") or None,
        opening=headers.get("Opening") or None,
        termination=headers.get("Termination") or None,
        site=headers.get("Site") or headers.get("Link") or None,
    )


_GAME_SPLIT = re.compile(r"\n\s*\n(?=\[Event )")


def iter_pgn_chunks(text: str) -> Iterator[str]:
    """Çok oyunlu PGN metnini oyun oyun böler (ayrıştırmadan)."""
    text = text.replace("\r\n", "\n").lstrip("﻿")
    for chunk in _GAME_SPLIT.split(text):
        if chunk.strip():
            yield chunk


def parse_pgn_text(
    text: str,
    matcher: PlayerMatcher,
    source: str,
    tempo_hint: str | None = None,
    stats: Counter | None = None,
    prefilter: bool = True,
) -> list[GameRecord]:
    """PGN metnindeki hedef oyuncuya ait oyunları kayıtlara çevirir.

    `prefilter` açıkken, oyuncunun adı geçmeyen oyunlar hiç ayrıştırılmaz;
    TWIC gibi binlerce oyunluk arşivlerde bu büyük hız kazandırır.
    """
    stats = stats if stats is not None else Counter()
    records: list[GameRecord] = []
    for chunk in iter_pgn_chunks(text):
        stats["seen"] += 1
        if prefilter:
            head = chunk[: chunk.find("\n\n")] if "\n\n" in chunk else chunk
            if not matcher.maybe_in(_normalize(head) if not matcher.exact else head.casefold()):
                stats["skipped_prefilter"] += 1
                continue
        try:
            game = chess.pgn.read_game(io.StringIO(chunk))
        except Exception:  # bozuk PGN tek oyunu düşürür, tüm işi değil
            stats["dropped_parse_error"] += 1
            continue
        if game is None:
            continue
        record = record_from_game(game, matcher, source, tempo_hint, stats)
        if record is not None:
            records.append(record)
    return records


def dedupe(records: Iterable[GameRecord], stats: Counter | None = None) -> list[GameRecord]:
    """Aynı oyunun kopyalarını eler (aynı kimlik ya da aynı gün + aynı hamleler)."""
    stats = stats if stats is not None else Counter()
    seen_ids: set[str] = set()
    seen_moves: set[tuple[str, str]] = set()
    out: list[GameRecord] = []
    for record in records:
        move_key = ((record.date or "")[:10], " ".join(record.moves))
        if record.game_id in seen_ids or move_key in seen_moves:
            stats["dropped_duplicate"] += 1
            continue
        seen_ids.add(record.game_id)
        seen_moves.add(move_key)
        out.append(record)
    return out


def save_jsonl(records: Iterable[GameRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.to_json() + "\n")


def load_jsonl(path: str | Path) -> list[GameRecord]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [GameRecord.from_json(line) for line in fh if line.strip()]
