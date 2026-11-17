"""Veri toplama orkestrasyonu: kaynakları birleştirir, temizler, tempoya ayırır.

Bu modül Kaggle/Colab not defterinin içinden çağrılır; yerel makinede büyük
indirme yapılmaz.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import FULL_FINETUNE_GAMES, LOW_DATA_GAMES, NEIGHBOUR_TEMPOS, TEMPO_LABELS_TR, TEMPOS
from .games import GameRecord, PlayerMatcher, dedupe
from .http import NotFound, RateLimitedFetcher


@dataclass
class CollectConfig:
    display_name: str                       # raporda ve sitede görünen ad
    tempo: str = "blitz"                    # hazırlanılan tempo
    lichess_user: str | None = None
    chesscom_user: str | None = None
    otb_names: list[str] = field(default_factory=list)   # "Soyad, Ad" biçimi önerilir
    fide_id: str | None = None
    use_twic: bool = False
    twic_issues: int = 150
    pgn_paths: list[str] = field(default_factory=list)
    max_games: int | None = 20000
    # Eğitime ek veri olarak katılacak diğer tempolar. None: komşu tempolar
    # (ör. rapid için klasik ve blitz). []: yalnızca seçilen tempo.
    extra_tempos: list[str] | None = None

    def __post_init__(self):
        if self.tempo not in TEMPOS:
            raise ValueError(f"tempo şunlardan biri olmalı: {', '.join(TEMPOS)}")
        if self.extra_tempos is None:
            self.extra_tempos = list(NEIGHBOUR_TEMPOS[self.tempo])
        unknown = [t for t in self.extra_tempos if t not in TEMPOS]
        if unknown:
            raise ValueError(f"bilinmeyen ek tempo: {unknown}")
        self.extra_tempos = [t for t in dict.fromkeys(self.extra_tempos) if t != self.tempo]
        if not (self.lichess_user or self.chesscom_user or self.use_twic or self.pgn_paths):
            raise ValueError("En az bir kaynak gerekli: lichess_user, chesscom_user, use_twic ya da pgn_paths")
        if (self.use_twic or self.pgn_paths) and not (self.otb_names or self.fide_id):
            raise ValueError("TWIC / PGN kaynakları için otb_names ya da fide_id gerekli")


@dataclass
class CollectResult:
    games: list[GameRecord]                 # seçilen tempodaki temiz oyunlar
    counts_by_tempo: dict[str, int]         # oyuncunun her tempodaki oyun sayısı
    stats: dict[str, dict[str, int]]        # kaynak başına temizlik sayaçları
    warnings: list[str]
    extra_games: list[GameRecord] = field(default_factory=list)  # ek tempolardaki oyunlar (tarih seçimi sonra)
    events: list[dict] = field(default_factory=list)              # etkinlik -> tempo dökümü


_EVENT_NOISE = re.compile(
    r"\b\d+(st|nd|rd|th)?\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b(early|late)\b|[()#.,:/-]"
)


def event_family(event: str | None) -> str:
    """Etkinlik adını haftalık/yıllık tekrarlardan arındırır.

    "Titled Tue 2nd Jan Early" ve "Titled Tue 9th Jan Late" -> "titled tue".
    """
    text = _EVENT_NOISE.sub(" ", (event or "?").casefold())
    return " ".join(text.split()) or "?"


def event_summary(records: list[GameRecord], top: int = 60) -> list[dict]:
    """Hangi etkinlik ailesi hangi tempoya, hangi kuralla konmuş: en kalabalıktan başlayarak."""
    groups: Counter = Counter()
    examples: dict[tuple, str] = {}
    for r in records:
        key = (event_family(r.event), r.tempo, r.tempo_source or "?", r.source)
        groups[key] += 1
        examples.setdefault(key, r.event or "?")
    return [
        {"family": fam, "tempo": tempo, "how": how, "source": source, "games": n,
         "example": examples[(fam, tempo, how, source)]}
        for (fam, tempo, how, source), n in groups.most_common(top)
    ]


def log_event_summary(events: list[dict], log=print, per_tempo: int = 12) -> None:
    """Not defteri çıktısı için: her tempoda en kalabalık etkinlikler."""
    how_tr = {"event_name": "etkinlik adı", "time_control": "süre ayarı", "site": "site", "?": "?"}
    log("[veri] Etkinliklerin tempoya dağılımı (tempo nasıl belirlendi, kontrol edin):")
    for tempo in TEMPOS:
        rows = [e for e in events if e["tempo"] == tempo][:per_tempo]
        if not rows:
            continue
        log(f"  {TEMPO_LABELS_TR[tempo]}:")
        for e in rows:
            log(f"    {e['games']:>5}  {e['example'][:48]:<48}  ({how_tr.get(e['how'], e['how'])}, {e['source']})")


def data_warnings(tempo: str, counts: dict[str, int], extra_tempos: list[str] | None = None) -> list[str]:
    """Seçilen tempoda veri azsa kullanıcıya gösterilecek uyarılar."""
    n = counts.get(tempo, 0)
    label = TEMPO_LABELS_TR[tempo]
    out: list[str] = []
    if n < LOW_DATA_GAMES:
        out.append(
            f"Bu oyuncunun {label} tempoda yalnızca {n} oyunu var. "
            f"Bot yine eğitilir ama benzerlik skoruna güven düşük olacak."
        )
    elif n < FULL_FINETUNE_GAMES:
        out.append(
            f"{label.capitalize()} tempoda {n} oyun var: sınırlı veri. "
            f"Model hafif ince ayarla eğitilecek, güven seviyesi orta."
        )
    used = [t for t in (extra_tempos or []) if counts.get(t, 0)]
    if used:
        parts = ", ".join(f"{TEMPO_LABELS_TR[t]} {counts[t]}" for t in used)
        out.append(
            f"Eğitime ek veri olarak diğer tempolardaki oyunlar da katıldı ({parts}). "
            f"Benzerlik yine yalnızca {label} oyunlarla ölçülür."
        )
    richer = sorted(
        ((c, t) for t, c in counts.items() if t != tempo and t not in used and c > max(n, LOW_DATA_GAMES)),
        reverse=True,
    )
    if n < FULL_FINETUNE_GAMES and richer:
        c, t = richer[0]
        out.append(
            f"{TEMPO_LABELS_TR.get(t, t).capitalize()} tempoda {c} oyun var. "
            f"O tempoda eğitilen bot daha güvenilir olur, ama farklı tempoda oyun tarzı değişebilir."
        )
    return out


def collect(
    cfg: CollectConfig,
    fetcher: RateLimitedFetcher,
    lichess_token: str | None = None,
    log=print,
) -> CollectResult:
    from .sources import chesscom, lichess, pgnfile, twic

    all_records: list[GameRecord] = []
    stats: dict[str, Counter] = {}
    counts: Counter = Counter()

    if cfg.lichess_user:
        stats["lichess"] = Counter()
        log(f"[veri] Lichess: {cfg.lichess_user} ({cfg.tempo}) indiriliyor...")
        records = lichess.fetch_games(
            cfg.lichess_user, [cfg.tempo, *cfg.extra_tempos], fetcher, max_games=cfg.max_games,
            token=lichess_token, stats=stats["lichess"],
        )
        all_records.extend(records)
        try:
            # İndirilmeyen tempoların sayısı profil özetinden gelir.
            for tempo, n in lichess.tempo_counts(cfg.lichess_user, fetcher, lichess_token).items():
                if tempo != cfg.tempo and tempo not in cfg.extra_tempos:
                    counts[tempo] += n
        except Exception as exc:
            log(f"[veri] Lichess tempo sayıları alınamadı: {exc}")

    if cfg.chesscom_user:
        stats["chesscom"] = Counter()
        log(f"[veri] Chess.com: {cfg.chesscom_user} arşivleri indiriliyor...")
        try:
            all_records.extend(chesscom.fetch_games(cfg.chesscom_user, fetcher, stats=stats["chesscom"]))
        except NotFound:
            # Bazı hesapların (ör. kapalı ya da gizli arşivler) oyun arşivi API'de yoktur.
            stats["chesscom"]["archives_unavailable"] = 1
            log(f"[UYARI] Chess.com: {cfg.chesscom_user} için oyun arşivi API'de bulunamadı; bu kaynak atlanıyor.")

    if cfg.use_twic or cfg.pgn_paths:
        matcher = PlayerMatcher(cfg.otb_names, fide_id=cfg.fide_id)
        if cfg.use_twic:
            stats["twic"] = Counter()
            log(f"[veri] TWIC: son {cfg.twic_issues} sayı taranıyor...")
            all_records.extend(
                twic.fetch_games(matcher, fetcher, issues=cfg.twic_issues, stats=stats["twic"], log=log)
            )
        if cfg.pgn_paths:
            stats["pgn"] = Counter()
            all_records.extend(pgnfile.load_paths(list(cfg.pgn_paths), matcher, stats=stats["pgn"]))

    stats["all"] = Counter()
    unique = dedupe(all_records, stats["all"])
    counts.update(r.tempo for r in unique)

    selected = sorted((r for r in unique if r.tempo == cfg.tempo), key=lambda r: r.sort_key)
    if cfg.max_games and len(selected) > cfg.max_games:
        stats["all"]["dropped_over_max_games"] = len(selected) - cfg.max_games
        selected = selected[-cfg.max_games:]  # en yeniler kalır

    extra = sorted((r for r in unique if r.tempo in cfg.extra_tempos), key=lambda r: r.sort_key)
    if cfg.max_games and len(extra) > cfg.max_games:
        extra = extra[-cfg.max_games:]

    counts_dict = {t: int(counts.get(t, 0)) for t in TEMPOS}
    counts_dict[cfg.tempo] = len(selected)
    warnings = data_warnings(cfg.tempo, counts_dict, cfg.extra_tempos)
    events = event_summary(unique)
    if any(r.source in ("twic", "pgn") for r in unique):
        log_event_summary(events, log)
    dropped_unknown = sum(v.get("dropped_unknown_tempo", 0) for v in stats.values())
    if dropped_unknown:
        log(f"[veri] Temposu belirlenemeyen {dropped_unknown} çevrim içi oyun atıldı.")
    for w in warnings:
        log(f"[UYARI] {w}")
    log(f"[veri] {cfg.tempo}: {len(selected)} temiz oyun; ek tempolar {cfg.extra_tempos}: {len(extra)} oyun. "
        f"Tüm tempolar: {counts_dict}")
    return CollectResult(
        games=selected,
        counts_by_tempo=counts_dict,
        stats={k: dict(v) for k, v in stats.items()},
        warnings=warnings,
        extra_games=extra,
        events=events,
    )


def cache_dir_default(root: str | Path) -> Path:
    path = Path(root) / "cache" / "http"
    path.mkdir(parents=True, exist_ok=True)
    return path
