"""Veri toplama orkestrasyonu: kaynakları birleştirir, temizler, tempoya ayırır.

Bu modül Kaggle/Colab not defterinin içinden çağrılır; yerel makinede büyük
indirme yapılmaz.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import FULL_FINETUNE_GAMES, LOW_DATA_GAMES, TEMPO_LABELS_TR, TEMPOS
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

    def __post_init__(self):
        if self.tempo not in TEMPOS:
            raise ValueError(f"tempo şunlardan biri olmalı: {', '.join(TEMPOS)}")
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


def data_warnings(tempo: str, counts: dict[str, int]) -> list[str]:
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
    richer = sorted(((c, t) for t, c in counts.items() if t != tempo and c > max(n, LOW_DATA_GAMES)), reverse=True)
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
            cfg.lichess_user, cfg.tempo, fetcher, max_games=cfg.max_games,
            token=lichess_token, stats=stats["lichess"],
        )
        all_records.extend(records)
        try:
            # Lichess'ten yalnızca seçilen tempo indirilir; diğer tempoların
            # sayısı profil özetinden gelir.
            for tempo, n in lichess.tempo_counts(cfg.lichess_user, fetcher, lichess_token).items():
                if tempo != cfg.tempo:
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

    counts_dict = {t: int(counts.get(t, 0)) for t in TEMPOS}
    counts_dict[cfg.tempo] = len(selected)
    warnings = data_warnings(cfg.tempo, counts_dict)
    for w in warnings:
        log(f"[UYARI] {w}")
    log(f"[veri] {cfg.tempo}: {len(selected)} temiz oyun. Tüm tempolar: {counts_dict}")
    return CollectResult(
        games=selected,
        counts_by_tempo=counts_dict,
        stats={k: dict(v) for k, v in stats.items()},
        warnings=warnings,
    )


def cache_dir_default(root: str | Path) -> Path:
    path = Path(root) / "cache" / "http"
    path.mkdir(parents=True, exist_ok=True)
    return path
