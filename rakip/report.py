"""Rakip raporu (stil profili): eğitim sırasında JSON olarak üretilir, site Hub'dan okur.

Rapor bir ölçüm değil, bir tanıtımdır; bu yüzden oyuncunun TÜM oyunlarını
(en yeniler dahil) kullanır. Turnuva hazırlığında en güncel repertuvar en
değerli bilgidir. Botun kendisi ve benzerlik skoru ise test oyunlarını görmez.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Iterable

import chess

from .book import top_lines
from .config import TEMPO_LABELS_TR, parse_time_control
from .games import GameRecord
from .similarity import ERROR_THRESHOLD, PHASE_LABELS_TR, PHASES

BLUNDER_THRESHOLD = 20.0
MIN_TAG_POSITIONS = 30


# --------------------------------------------------------------------------
# Sonuçlar ve açılışlar
# --------------------------------------------------------------------------

def results_summary(games: Iterable[GameRecord]) -> dict:
    out = {}
    games = list(games)
    for key, subset in (
        ("all", games),
        ("white", [g for g in games if g.player_color == "white"]),
        ("black", [g for g in games if g.player_color == "black"]),
    ):
        scored = [g.result for g in subset if g.result is not None]
        out[key] = {
            "games": len(subset),
            "wins": sum(1 for r in scored if r == 1.0),
            "draws": sum(1 for r in scored if r == 0.5),
            "losses": sum(1 for r in scored if r == 0.0),
            "score": sum(scored) / len(scored) if scored else None,
        }
    return out


# --------------------------------------------------------------------------
# Hata dağılımı
# --------------------------------------------------------------------------

def phase_errors(details: list[dict]) -> dict:
    """Oyun aşamasına göre hata oranları (Stockfish değerlendirmesinden)."""
    rows = [d for d in details if d.get("loss") is not None and d["loss"] == d["loss"]]
    out = {"available": bool(rows), "n": len(rows), "phases": {}}
    if not rows:
        return out
    worst, worst_rate = None, -1.0
    for phase in PHASES:
        sub = [d["loss"] for d in rows if d["phase"] == phase]
        if not sub:
            out["phases"][phase] = {"n": 0}
            continue
        rate = sum(1 for x in sub if x >= ERROR_THRESHOLD) / len(sub)
        out["phases"][phase] = {
            "n": len(sub),
            "label": PHASE_LABELS_TR[phase],
            "error_rate": rate,
            "blunder_rate": sum(1 for x in sub if x >= BLUNDER_THRESHOLD) / len(sub),
            "avg_loss": sum(sub) / len(sub),
        }
        if len(sub) >= MIN_TAG_POSITIONS and rate > worst_rate:
            worst, worst_rate = phase, rate
    out["worst_phase"] = worst
    if worst:
        out["text"] = (
            f"En çok {PHASE_LABELS_TR[worst]} aşamasında hata yapıyor: "
            f"her 100 hamleden yaklaşık {round(100 * worst_rate)} tanesi ciddi hata."
        )
    return out


# --------------------------------------------------------------------------
# Zaman kullanımı
# --------------------------------------------------------------------------

def _trouble_threshold(base: float) -> float:
    return max(10.0, min(60.0, base * 0.1))


def time_usage(games: Iterable[GameRecord], details: list[dict] | None = None) -> dict:
    """Saat bilgisi olan oyunlardan zaman kullanımı. Yoksa `available: False`."""
    buckets: dict[str, list[float]] = defaultdict(list)   # hamle aralığı -> hamle başına süre oranı
    trouble_games = trouble_losses = with_clock = 0
    first_trouble_moves: list[int] = []
    order = ["1-10", "11-20", "21-30", "31-40", "41+"]
    for game in games:
        tc = parse_time_control(game.time_control)
        if not game.has_clocks or tc is None or tc[0] <= 0:
            continue
        base, inc = tc
        with_clock += 1
        threshold = _trouble_threshold(base)
        in_trouble_at = None
        indices = list(game.player_ply_indices())
        previous = base
        for ply in indices:
            if ply >= len(game.clocks) or game.clocks[ply] is None:
                continue
            after = game.clocks[ply]
            spent = max(0.0, previous - after + inc)
            move_no = ply // 2 + 1
            bucket = order[min(4, (move_no - 1) // 10)]
            buckets[bucket].append(spent / base)
            if in_trouble_at is None and after < threshold and move_no <= 60:
                in_trouble_at = move_no
            previous = after
        if in_trouble_at is not None:
            trouble_games += 1
            first_trouble_moves.append(in_trouble_at)
            if game.result == 0.0:
                trouble_losses += 1
    if with_clock == 0:
        return {"available": False, "text": "Bu oyunlarda saat bilgisi yok (masa başı oyunlarda genellikle kaydedilmez)."}

    per_bucket = [
        {"moves": b, "share_of_base_per_move": statistics.mean(buckets[b]), "n": len(buckets[b])}
        for b in order if buckets[b]
    ]
    fastest = min(per_bucket, key=lambda x: x["share_of_base_per_move"]) if per_bucket else None
    slowest = max(per_bucket, key=lambda x: x["share_of_base_per_move"]) if per_bucket else None
    out = {
        "available": True,
        "games_with_clock": with_clock,
        "time_trouble_rate": trouble_games / with_clock,
        "time_trouble_loss_rate": (trouble_losses / trouble_games) if trouble_games else None,
        "median_trouble_move": int(statistics.median(first_trouble_moves)) if first_trouble_moves else None,
        "per_move_range": per_bucket,
        "slowest_range": slowest["moves"] if slowest else None,
        "fastest_range": fastest["moves"] if fastest else None,
    }
    text = f"Oyunlarının %{round(100 * out['time_trouble_rate'])} kadarında zaman sıkıntısına düşüyor"
    if out["median_trouble_move"]:
        text += f" (tipik olarak {out['median_trouble_move']}. hamle civarında)"
    text += "."
    if slowest and fastest and slowest is not fastest:
        text += f" En çok {slowest['moves']}. hamleler arasında düşünüyor, {fastest['moves']}. hamlelerde hızlanıyor."
    out["text"] = text

    if details:
        with_time = [d for d in details if d.get("clock") is not None and d.get("time_control")]
        low, normal = [], []
        for d in with_time:
            tc = parse_time_control(d["time_control"])
            if tc is None or tc[0] <= 0:
                continue
            (low if d["clock"] < _trouble_threshold(tc[0]) else normal).append(d["loss"])
        if len(low) >= MIN_TAG_POSITIONS and normal:
            out["error_rate_in_trouble"] = sum(1 for x in low if x >= ERROR_THRESHOLD) / len(low)
            out["error_rate_normal"] = sum(1 for x in normal if x >= ERROR_THRESHOLD) / len(normal)
    return out


# --------------------------------------------------------------------------
# Stil: taktik mi pozisyonel mi
# --------------------------------------------------------------------------

def style_profile(games: Iterable[GameRecord]) -> dict:
    """Karmaşık pozisyonlara girme eğilimi için kaba göstergeler.

    Bunlar motor değerlendirmesi değil, oyunların yapısından çıkan sayımlardır.
    Etiket yalnızca yön gösterir; kesin bir sınıflandırma değildir.
    """
    games = list(games)
    opposite = queen_trade = 0
    contact: list[float] = []
    lengths: list[int] = []
    counted = 0
    for game in games:
        board = chess.Board()
        castle = {chess.WHITE: None, chess.BLACK: None}
        captures_checks = window = 0
        queens_off_by_20 = False
        for ply, uci in enumerate(game.moves):
            move = chess.Move.from_uci(uci)
            if board.is_castling(move):
                castle[board.turn] = "k" if board.is_kingside_castling(move) else "q"
            if 20 <= ply < 60:
                window += 1
                if board.is_capture(move) or board.gives_check(move):
                    captures_checks += 1
            board.push(move)
            if ply == 39 and not board.pieces(chess.QUEEN, chess.WHITE) and not board.pieces(chess.QUEEN, chess.BLACK):
                queens_off_by_20 = True
        counted += 1
        lengths.append((len(game.moves) + 1) // 2)
        if castle[chess.WHITE] and castle[chess.BLACK] and castle[chess.WHITE] != castle[chess.BLACK]:
            opposite += 1
        if queens_off_by_20:
            queen_trade += 1
        if window >= 20:
            contact.append(captures_checks / window)
    if counted == 0:
        return {"available": False}
    opposite_rate = opposite / counted
    trade_rate = queen_trade / counted
    contact_rate = statistics.mean(contact) if contact else 0.0
    # 0-100 "karmaşıklık" göstergesi: ters rok ve temas yoğunluğu artırır, erken vezir değişimi azaltır.
    index = 50 + 150 * (opposite_rate - 0.10) + 200 * (contact_rate - 0.25) - 60 * (trade_rate - 0.20)
    index = max(0.0, min(100.0, index))
    label = "taktik / keskin" if index >= 60 else "pozisyonel / sakin" if index <= 40 else "dengeli"
    return {
        "available": True,
        "complexity_index": round(index),
        "label": label,
        "opposite_castling_rate": opposite_rate,
        "early_queen_trade_rate": trade_rate,
        "contact_rate": contact_rate,
        "avg_game_length": statistics.mean(lengths),
        "text": (
            f"Tarz göstergesi: {label}. Oyunlarının %{round(100 * opposite_rate)} kadarında ters taraflara rok var, "
            f"%{round(100 * trade_rate)} kadarında vezirler 20. hamleye kadar değişilmiş."
        ),
        "note": "Bu, oyun yapısından çıkan kaba bir göstergedir; kesin bir sınıflandırma değildir.",
    }


# --------------------------------------------------------------------------
# Zayıf noktalar
# --------------------------------------------------------------------------

def _endgame_type(board: chess.Board) -> str:
    def has(piece):
        return bool(board.pieces(piece, chess.WHITE) or board.pieces(piece, chess.BLACK))

    queens, rooks = has(chess.QUEEN), has(chess.ROOK)
    minors = has(chess.BISHOP) or has(chess.KNIGHT)
    if queens:
        return "Vezirli oyun sonları"
    if rooks and not minors:
        return "Kale oyun sonları"
    if rooks:
        return "Kale ve hafif taşlı oyun sonları"
    if minors:
        return "Hafif taş oyun sonları"
    return "Piyon oyun sonları"


def position_tags(detail: dict) -> list[str]:
    """Bir pozisyonun ait olduğu, satranççının tanıyacağı tipler."""
    board = chess.Board(detail["fen"])
    tags: list[str] = []
    phase = detail["phase"]
    queens_on = bool(board.pieces(chess.QUEEN, chess.WHITE) and board.pieces(chess.QUEEN, chess.BLACK))
    if phase == "endgame":
        tags.append(_endgame_type(board))
    elif phase == "middlegame":
        tags.append("Vezirli orta oyunlar" if queens_on else "Vezirsiz orta oyunlar")
        wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)
        if wk is not None and bk is not None:
            wf, bf = chess.square_file(wk), chess.square_file(bk)
            if (wf <= 2 and bf >= 5) or (wf >= 5 and bf <= 2):
                tags.append("Şahların ters kanatlarda olduğu pozisyonlar")
    wp = detail.get("wp_before")
    if wp is not None:
        if wp >= 75:
            tags.append("Üstün pozisyonu kazanca çevirme")
        elif wp <= 30:
            tags.append("Kötü pozisyonda savunma")
        elif 40 <= wp <= 60 and phase != "opening":
            tags.append("Dengeli pozisyonlar")
    clock, tc = detail.get("clock"), parse_time_control(detail.get("time_control"))
    if clock is not None and tc and tc[0] > 0 and clock < _trouble_threshold(tc[0]):
        tags.append("Zaman sıkıntısı")
    if board.is_check():
        tags.append("Şah altındaki pozisyonlar")
    return tags


def weaknesses(details: list[dict], top: int = 5, examples: int = 3) -> dict:
    """En çok puan kaybedilen pozisyon tipleri, somut örnek pozisyonlarla."""
    rows = [d for d in details if d.get("loss") is not None and d["loss"] == d["loss"]]
    if len(rows) < MIN_TAG_POSITIONS:
        return {"available": False, "items": []}
    overall = sum(d["loss"] for d in rows) / len(rows)
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for d in rows:
        for tag in position_tags(d):
            by_tag[tag].append(d)
    items = []
    for tag, members in by_tag.items():
        if len(members) < MIN_TAG_POSITIONS:
            continue
        avg = sum(d["loss"] for d in members) / len(members)
        rate = sum(1 for d in members if d["loss"] >= ERROR_THRESHOLD) / len(members)
        worst = sorted(members, key=lambda d: -d["loss"])
        picked, seen_games = [], set()
        for d in worst:
            if d["game_id"] in seen_games or d["loss"] < ERROR_THRESHOLD:
                continue
            seen_games.add(d["game_id"])
            picked.append({
                "fen": d["fen"], "played": d["played"], "played_san": d["played_san"],
                "best": d["best"], "best_san": d["best_san"], "loss": round(d["loss"], 1),
                "move_number": d["ply"] // 2 + 1, "date": (d.get("date") or "")[:10] or None,
                "opponent": d.get("opponent"), "site": d.get("site"),
            })
            if len(picked) >= examples:
                break
        items.append({
            "type": tag, "n": len(members), "avg_loss": avg, "error_rate": rate,
            "excess": avg - overall, "examples": picked,
            "text": f"Bu tip pozisyonlarda her 100 hamleden yaklaşık {round(100 * rate)} tanesi ciddi hata.",
        })
    items = [i for i in items if i["excess"] > 0]
    items.sort(key=lambda i: -i["excess"])
    return {"available": bool(items), "overall_avg_loss": overall, "items": items[:top]}


# --------------------------------------------------------------------------
# Rapor
# --------------------------------------------------------------------------

def build_report(
    *,
    player: str,
    tempo: str,
    games: list[GameRecord],
    details: list[dict],
    counts_by_tempo: dict[str, int] | None = None,
    warnings: list[str] | None = None,
    sources: dict | None = None,
) -> dict:
    dates = sorted(g.date[:10] for g in games if g.date)
    ratings = [g.player_rating for g in games if g.player_rating]
    return {
        "version": 1,
        "player": player,
        "tempo": tempo,
        "tempo_label": TEMPO_LABELS_TR.get(tempo, tempo),
        "summary": {
            "games": len(games),
            "first_game": dates[0] if dates else None,
            "last_game": dates[-1] if dates else None,
            "rating_latest": next((g.player_rating for g in reversed(games) if g.player_rating), None),
            "rating_peak": max(ratings) if ratings else None,
            "counts_by_tempo": counts_by_tempo or {},
            "sources": dict(Counter(g.source for g in games)),
        },
        "warnings": warnings or [],
        "results": results_summary(games),
        "openings": {
            "white": top_lines(games, "white", depth_plies=8, top=5),
            "black": top_lines(games, "black", depth_plies=8, top=5),
        },
        "phase_errors": phase_errors(details),
        "time": time_usage(games, details),
        "style": style_profile(games),
        "weaknesses": weaknesses(details),
        "data_cleaning": sources or {},
        "analysed_positions": len(details),
    }
