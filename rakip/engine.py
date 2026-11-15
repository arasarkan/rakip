"""Stockfish ile pozisyon değerlendirme (Kaggle/Colab'da CPU üzerinde).

Her pozisyon için: en iyi hamle ve değerlendirmesi, ayrıca verilen hamlelerin
(rakibin oynadığı, botun oynayacağı vb.) değerlendirmesi. Bunlardan "kazanma
şansı kaybı" hesaplanır; hata profili ve rakip raporu bu kayıplara dayanır.

`engine_pass`, değerlendiriciyi parametre olarak alır; testlerde sahte bir
değerlendirici verilir, böylece mantık Stockfish olmadan doğrulanır.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import chess
import numpy as np

from .encoding import index_move, move_index
from .features import SampleSet
from .games import GameRecord
from .similarity import phases_from_codes, win_percent


@dataclass
class EvalJob:
    key: int                 # örnek satırı
    fen: str
    moves: list[str]         # değerlendirilecek hamleler (UCI)


@dataclass
class EvalResult:
    key: int
    best: str | None = None
    cp_best: float = float("nan")          # hamle sırası olan tarafın gözünden, santipiyon
    cp_moves: dict[str, float] = field(default_factory=dict)


Evaluator = Callable[[list[EvalJob]], list[EvalResult]]


def find_stockfish() -> str | None:
    for candidate in (shutil.which("stockfish"), "/usr/games/stockfish", "/usr/bin/stockfish", "/usr/local/bin/stockfish"):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def ensure_stockfish(log=print) -> str | None:
    """Stockfish yoksa apt ile kurmayı dener (Kaggle/Colab kök yetkisiyle çalışır)."""
    path = find_stockfish()
    if path:
        return path
    log("[stockfish] kuruluyor (apt-get install stockfish)...")
    try:
        subprocess.run(["apt-get", "install", "-y", "-qq", "stockfish"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    except Exception as exc:
        log(f"[stockfish] kurulamadı: {exc}")
    return find_stockfish()


def _score_cp(info, turn: chess.Color) -> float:
    return float(info["score"].pov(turn).score(mate_score=10000))


def stockfish_evaluator(
    engine_path: str,
    nodes: int = 40000,
    workers: int | None = None,
    time_budget_seconds: float | None = None,
    log=print,
) -> Evaluator:
    """Gerçek Stockfish değerlendiricisi. Sabit düğüm sayısı kullanır (süre değil),
    böylece sonuçlar makineden makineye değişmez."""
    import chess.engine

    workers = workers or max(1, (os.cpu_count() or 2))

    def run(jobs: list[EvalJob]) -> list[EvalResult]:
        results: list[EvalResult | None] = [None] * len(jobs)
        cursor = {"next": 0, "done": 0}
        lock = threading.Lock()
        started = time.time()
        deadline = None if time_budget_seconds is None else started + time_budget_seconds

        def worker() -> None:
            engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            try:
                engine.configure({"Threads": 1, "Hash": 64})
                while True:
                    with lock:
                        i = cursor["next"]
                        if i >= len(jobs) or (deadline is not None and time.time() > deadline):
                            return
                        cursor["next"] += 1
                    job = jobs[i]
                    result = EvalResult(key=job.key)
                    try:
                        board = chess.Board(job.fen)
                        limit = chess.engine.Limit(nodes=nodes)
                        info = engine.analyse(board, limit)
                        result.cp_best = _score_cp(info, board.turn)
                        pv = info.get("pv") or []
                        result.best = pv[0].uci() if pv else None
                        others = []
                        for uci in dict.fromkeys(job.moves):
                            if uci == result.best:
                                result.cp_moves[uci] = result.cp_best
                            else:
                                move = chess.Move.from_uci(uci)
                                if move in board.legal_moves:
                                    others.append(move)
                        if others:
                            infos = engine.analyse(board, limit, multipv=len(others), root_moves=others)
                            for item in infos:
                                if item.get("pv"):
                                    result.cp_moves[item["pv"][0].uci()] = _score_cp(item, board.turn)
                    except chess.engine.EngineTerminatedError:
                        raise
                    except Exception:
                        pass  # tek pozisyon atlanır
                    results[i] = result
                    with lock:
                        cursor["done"] += 1
                        done = cursor["done"]
                    if done in (50, 500) or done % 2000 == 0:
                        rate = done / max(1e-6, time.time() - started)
                        remaining = (len(jobs) - done) / max(1e-6, rate)
                        log(f"[stockfish] {done}/{len(jobs)} pozisyon, kalan tahmini süre {remaining / 60:.0f} dk")
            finally:
                try:
                    engine.quit()
                except Exception:
                    pass

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        skipped = sum(1 for r in results if r is None)
        if skipped:
            log(f"[stockfish] süre bütçesi nedeniyle {skipped} pozisyon değerlendirilmedi.")
        return [r if r is not None else EvalResult(key=jobs[i].key) for i, r in enumerate(results)]

    return run


def boards_for_rows(games: list[GameRecord], samples: SampleSet, rows: Iterable[int]) -> dict[int, chess.Board]:
    """Örnek satırlarının pozisyonlarını oyunları yeniden oynayarak üretir."""
    wanted: dict[int, dict[int, int]] = {}
    for row in rows:
        wanted.setdefault(int(samples.game_index[row]), {})[int(samples.ply[row])] = int(row)
    boards: dict[int, chess.Board] = {}
    for gi, plies in wanted.items():
        board = chess.Board()
        last = max(plies)
        for ply, uci in enumerate(games[gi].moves):
            if ply in plies:
                boards[plies[ply]] = board.copy(stack=False)
            if ply >= last:
                break
            board.push(chess.Move.from_uci(uci))
    return boards


@dataclass
class EnginePass:
    rows: np.ndarray                       # değerlendirilen örnek satırları
    sf_best: np.ndarray                    # (N,) Stockfish'in en iyi hamlesi (sözlük indeksi), yoksa -1
    loss_player: np.ndarray                # (N,) kazanma şansı kaybı, yoksa nan
    loss: dict[str, np.ndarray]            # ad -> (N,) kayıp (ör. "bot", "generic")
    wp_before: np.ndarray                  # (N,) hamleden önce kazanma şansı (en iyi hamleyle)
    phases: np.ndarray                     # (N,) oyun aşaması
    details: list[dict]                    # rapor için pozisyon ayrıntıları


def choose_rows(mask: np.ndarray, limit: int | None, seed: int = 0) -> np.ndarray:
    rows = np.flatnonzero(mask)
    if limit is not None and rows.size > limit:
        rows = np.sort(np.random.default_rng(seed).choice(rows, size=limit, replace=False))
    return rows


def engine_pass(
    games: list[GameRecord],
    samples: SampleSet,
    rows: np.ndarray,
    evaluator: Evaluator,
    move_sets: dict[str, np.ndarray] | None = None,
) -> EnginePass:
    """Seçilen satırlarda rakibin hamlesini ve verilen ek hamleleri değerlendirir.

    `move_sets`: ad -> (N,) sözlük indeksleri (ör. botun örneklenen hamleleri).
    """
    move_sets = move_sets or {}
    n = len(samples)
    boards = boards_for_rows(games, samples, rows)
    jobs: list[EvalJob] = []
    played: dict[int, dict[str, str]] = {}
    for row in rows:
        row = int(row)
        board = boards[row]
        names = {"player": index_move(int(samples.target[row]), board.turn).uci()}
        for name, indices in move_sets.items():
            names[name] = index_move(int(indices[row]), board.turn).uci()
        played[row] = names
        jobs.append(EvalJob(key=row, fen=board.fen(), moves=list(dict.fromkeys(names.values()))))

    sf_best = np.full(n, -1, dtype=np.int32)
    loss_player = np.full(n, np.nan, dtype=np.float32)
    losses = {name: np.full(n, np.nan, dtype=np.float32) for name in move_sets}
    wp_before = np.full(n, np.nan, dtype=np.float32)
    phases = phases_from_codes(samples.codes[samples.hist[:, -1]], samples.ply)
    details: list[dict] = []

    for result in evaluator(jobs):
        row = result.key
        if result.best is None or np.isnan(result.cp_best):
            continue
        board = boards[row]
        best_move = chess.Move.from_uci(result.best)
        sf_best[row] = move_index(best_move, board.turn)
        wp_best = float(win_percent(result.cp_best))
        wp_before[row] = wp_best

        def loss_of(uci: str) -> float:
            if uci == result.best:
                return 0.0
            cp = result.cp_moves.get(uci)
            if cp is None:
                return float("nan")
            return max(0.0, wp_best - float(win_percent(cp)))

        names = played[row]
        loss_player[row] = loss_of(names["player"])
        for name in move_sets:
            losses[name][row] = loss_of(names[name])
        game = games[int(samples.game_index[row])]
        details.append({
            "row": row,
            "fen": board.fen(),
            "ply": int(samples.ply[row]),
            "phase": str(phases[row]),
            "clock": None if np.isnan(samples.clock[row]) else float(samples.clock[row]),
            "played": names["player"],
            "played_san": board.san(chess.Move.from_uci(names["player"])),
            "best": result.best,
            "best_san": board.san(best_move),
            "wp_before": wp_best,
            "loss": float(loss_player[row]),
            "game_id": game.game_id,
            "date": game.date,
            "opponent": game.opponent_name,
            "site": game.site,
            "time_control": game.time_control,
        })
    return EnginePass(rows=np.asarray(rows), sf_best=sf_best, loss_player=loss_player, loss=losses,
                      wp_before=wp_before, phases=phases, details=details)
