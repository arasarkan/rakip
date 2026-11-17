"""İnce ayar: süre tahmini, erken durdurma, her epoch'ta Hub'a checkpoint, kaldığı yerden devam.

Bu dosya torch gerektirir ve yalnızca Kaggle/Colab'da çalışır.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .budget import REGIME_DEFAULTS, TIME_BUDGET_HOURS, BudgetPlan, choose_regime, format_duration, plan_budget
from .config import BASE_MODEL_FILE, BASE_MODEL_REPO
from .encoding import N_MOVES
from .features import SampleSet
from .model import MAIA3_5M, RakipModel, load_base, set_trainable
from .similarity import Predictions

STATE_FILE = "checkpoints/state.pt"
BEST_FILE = "checkpoints/best.pt"


@dataclass
class TrainConfig:
    regime: str = "auto"              # auto | embedding | light | full
    max_epochs: int | None = None     # None: rejimin varsayılanı
    batch_size: int = 256
    lr: float | None = None
    lr_player: float | None = None
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    patience: int = 2                 # doğrulama kaybı bu kadar epoch iyileşmezse dur
    min_delta: float = 1e-4
    time_budget_hours: float = TIME_BUDGET_HOURS
    reserve_seconds: float = 3600.0   # Stockfish değerlendirmesi ve dışa aktarım için ayrılan süre
    seed: int = 0

    def resolved(self, n_games: int) -> "TrainConfig":
        regime = choose_regime(n_games) if self.regime == "auto" else self.regime
        epochs, lr, lr_player = REGIME_DEFAULTS[regime]
        return TrainConfig(
            regime=regime,
            max_epochs=self.max_epochs or epochs,
            batch_size=self.batch_size,
            lr=self.lr or lr,
            lr_player=self.lr_player or lr_player,
            weight_decay=self.weight_decay,
            warmup_fraction=self.warmup_fraction,
            patience=self.patience,
            min_delta=self.min_delta,
            time_budget_hours=self.time_budget_hours,
            reserve_seconds=self.reserve_seconds,
            seed=self.seed,
        )


@dataclass
class FitResult:
    history: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    stopped_early: bool = False
    resumed_from_epoch: int | None = None
    plan: dict | None = None
    regime: str = ""
    trainable_parameters: int = 0
    seconds: float = 0.0


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_base_from_hub(token: str | None = None, device: torch.device | None = None) -> RakipModel:
    """Maia3-5M ağırlıklarını Hub'dan indirir (21 MB) ve yükler."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(BASE_MODEL_REPO, BASE_MODEL_FILE, token=token)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = load_base(state, MAIA3_5M)
    return model.to(device or pick_device())


class DeviceSamples:
    """Örnekleri cihaza taşır; (B, 64, 96) girdiyi toplu iş anında üretir."""

    def __init__(self, samples: SampleSet, device: torch.device, self_elo: float | None = None):
        self.n = len(samples)
        self.device = device
        self.codes = torch.from_numpy(samples.codes.astype(np.uint8)).to(device)
        self.hist = torch.from_numpy(samples.hist.astype(np.int64)).to(device)
        self.target = torch.from_numpy(samples.target.astype(np.int64)).to(device)
        self.legal = torch.from_numpy(samples.legal).to(device)
        self.oppo_elo = torch.from_numpy(samples.oppo_elo).to(device)
        self.weight = torch.from_numpy(samples.weights()).to(device)
        if self_elo is None:
            self.self_elo = torch.from_numpy(samples.self_elo).to(device)
        else:
            self.self_elo = torch.full((self.n,), float(self_elo), device=device)
        self._shifts = torch.arange(7, -1, -1, device=device, dtype=torch.uint8)

    def batch(self, rows: torch.Tensor):
        boards = self.codes[self.hist[rows]].long()                       # (B, 8, 64)
        tokens = F.one_hot(boards, 13)[..., 1:].float()                   # (B, 8, 64, 12)
        tokens = tokens.permute(0, 2, 1, 3).reshape(rows.shape[0], 64, 96)
        packed = self.legal[rows]                                         # (B, 544) uint8
        legal = ((packed.unsqueeze(-1) >> self._shifts) & 1).bool().reshape(rows.shape[0], N_MOVES)
        return tokens, self.self_elo[rows], self.oppo_elo[rows], legal, self.target[rows]


def masked_logits(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return logits.float().masked_fill(~legal, -1e9)


@torch.no_grad()
def predict(
    model: RakipModel,
    samples: SampleSet,
    device: torch.device,
    self_elo: float | None = None,
    use_player: bool = True,
    batch_size: int = 1024,
    temperature: float = 1.0,
    seed: int = 0,
) -> Predictions:
    """Her örnek için modelin ilk tahmini, oynanan hamleye verdiği olasılık ve
    (botun gerçekte yapacağı gibi) dağılımdan örneklenen bir hamle."""
    model.eval()
    data = DeviceSamples(samples, device, self_elo)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample = np.zeros(data.n, dtype=np.int32)
    top1 = np.zeros(data.n, dtype=np.int32)
    p_target = np.zeros(data.n, dtype=np.float32)
    p_top1 = np.zeros(data.n, dtype=np.float32)
    for start in range(0, data.n, batch_size):
        rows = torch.arange(start, min(data.n, start + batch_size), device=device)
        tokens, se, oe, legal, target = data.batch(rows)
        logits, _ = model(tokens, se, oe, use_player=use_player)
        masked = masked_logits(logits, legal)
        probs = torch.softmax(masked, dim=-1)
        sampling = probs if temperature == 1.0 else torch.softmax(masked / max(temperature, 1e-3), dim=-1)
        drawn = torch.multinomial(sampling.cpu(), 1, generator=generator).squeeze(1)
        sample[start:start + rows.shape[0]] = drawn.numpy()
        best_p, best = probs.max(dim=-1)
        top1[start:start + rows.shape[0]] = best.cpu().numpy()
        p_top1[start:start + rows.shape[0]] = best_p.cpu().numpy()
        p_target[start:start + rows.shape[0]] = probs.gather(1, target.unsqueeze(1)).squeeze(1).cpu().numpy()
    return Predictions(top1=top1, p_target=p_target, p_top1=p_top1, sample=sample)


def summarize(pred: Predictions, samples: SampleSet, rows: np.ndarray | None = None) -> dict:
    rows = samples.measured if rows is None else rows
    if rows.sum() == 0:
        return {"n": 0, "top1": float("nan"), "nll": float("nan")}
    hit = pred.top1[rows] == samples.target[rows]
    nll = -np.log(np.clip(pred.p_target[rows], 1e-9, 1.0))
    return {"n": int(rows.sum()), "top1": float(hit.mean()), "nll": float(nll.mean())}


def calibrate_elo(
    model: RakipModel, val: SampleSet, device: torch.device, nominal: int, log=print
) -> tuple[int, list[dict]]:
    """Genel modelin bu oyuncuyu en iyi tahmin ettiği reytingi doğrulama setinde arar.

    Reyting ölçekleri (FIDE, Chess.com, Lichess) birbirini tutmaz; bu arama hem
    ince ayarın başlangıç noktasını hem de dürüst bir karşılaştırma tabanını verir.
    Test seti bu seçimde kullanılmaz.
    """
    candidates = sorted({int(min(3400, max(600, nominal + d))) for d in (-600, -400, -200, 0, 200, 400, 600)})
    table = []
    for elo in candidates:
        stats = summarize(predict(model, val, device, self_elo=elo, use_player=False), val)
        table.append({"elo": elo, **stats})
    usable = [row for row in table if row["n"] > 0]
    best = min(usable, key=lambda row: row["nll"])["elo"] if usable else nominal
    for row in table:
        mark = " <- seçildi" if row["elo"] == best else ""
        log(f"[reyting] {row['elo']}: isabet %{100 * row['top1']:.1f}, kayıp {row['nll']:.3f}{mark}")
    return best, table


def run_signature(split_manifest: dict, cfg: TrainConfig, self_elo: float) -> str:
    """Checkpoint'in hangi veri ve ayarla üretildiğinin özeti; uyuşmazsa devam edilmez."""
    payload = {
        "digest": split_manifest.get("digest"),
        "extra": split_manifest.get("extra_training"),
        "cfg": asdict(cfg),
        "self_elo": float(self_elo),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _lr_factor(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _benchmark(model, data: DeviceSamples, batch_size: int, device) -> tuple[float, float]:
    """Bir eğitim adımının ve bir değerlendirme adımının süresi (ağırlıklar değişmez)."""
    n = min(data.n, batch_size)
    rows = torch.arange(n, device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def train_step():
        tokens, se, oe, legal, target = data.batch(rows)
        logits, _ = model(tokens, se, oe)
        F.cross_entropy(masked_logits(logits, legal), target).backward()
        model.zero_grad(set_to_none=True)

    model.train()
    for _ in range(3):
        train_step()
    sync()
    start = time.perf_counter()
    for _ in range(10):
        train_step()
    sync()
    train_sec = (time.perf_counter() - start) / 10 * (batch_size / n)

    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(5):
            tokens, se, oe, legal, _ = data.batch(rows)
            model(tokens, se, oe)
        sync()
        eval_sec = (time.perf_counter() - start) / 5 * (batch_size / n)
    return train_sec, eval_sec


@torch.no_grad()
def _val_loss(model, data: DeviceSamples, rows_all: torch.Tensor, batch_size: int) -> tuple[float, float]:
    model.eval()
    loss_sum, hits, count = 0.0, 0, 0
    for start in range(0, rows_all.shape[0], batch_size * 2):
        rows = rows_all[start:start + batch_size * 2]
        tokens, se, oe, legal, target = data.batch(rows)
        logits = masked_logits(model(tokens, se, oe)[0], legal)
        loss_sum += F.cross_entropy(logits, target, reduction="sum").item()
        hits += (logits.argmax(dim=-1) == target).sum().item()
        count += rows.shape[0]
    return loss_sum / max(1, count), hits / max(1, count)


def fit(
    model: RakipModel,
    train: SampleSet,
    val: SampleSet,
    cfg: TrainConfig,
    device: torch.device,
    self_elo: float,
    signature: str,
    work_dir: str | Path,
    store=None,
    session_start: float | None = None,
    log=print,
) -> FitResult:
    """Modeli ince ayarlar ve en iyi doğrulama kaybına sahip ağırlıkları yükleyip döner.

    `cfg` çözülmüş olmalı (`TrainConfig.resolved`). `store` verilirse her epoch
    sonunda checkpoint Hub'a yüklenir ve açılışta oradan devam edilir.
    """
    work_dir = Path(work_dir)
    (work_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    started = time.time()
    session_start = session_start or started
    torch.manual_seed(cfg.seed)

    trainable = set_trainable(model, cfg.regime)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = FitResult(regime=cfg.regime, trainable_parameters=n_trainable)
    log(f"[eğitim] rejim: {cfg.regime}, eğitilen parametre: {n_trainable:,} ({len(trainable)} tensör)")

    train_data = DeviceSamples(train, device, self_elo)
    val_data = DeviceSamples(val, device, self_elo)
    val_rows = torch.from_numpy(np.flatnonzero(val.measured)).to(device)
    if val_rows.shape[0] == 0:
        val_rows = torch.arange(val_data.n, device=device)

    # --- süre tahmini ve bütçe -------------------------------------------
    train_sec, eval_sec = _benchmark(model, train_data, cfg.batch_size, device)
    elapsed = time.time() - session_start
    budget = cfg.time_budget_hours * 3600 - elapsed
    plan: BudgetPlan = plan_budget(
        n_train=train_data.n, n_val=int(val_rows.shape[0]), batch_size=cfg.batch_size,
        sec_per_train_step=train_sec, sec_per_eval_step=eval_sec * 2, max_epochs=cfg.max_epochs,
        budget_seconds=budget, fixed_overhead_seconds=cfg.reserve_seconds,
    )
    log("[eğitim] " + plan.describe())
    log(f"[eğitim] (tahmin, değerlendirme ve dışa aktarım için ayrılan {format_duration(cfg.reserve_seconds)} dahil)")
    result.plan = asdict(plan)
    # Örnekler tarihe göre sıralıdır; kısıtlama gerekiyorsa en yeniler tutulur.
    train_rows_all = torch.arange(train_data.n - plan.n_train, train_data.n, device=device)
    epochs, steps = plan.epochs, plan.steps_per_epoch
    total_steps = epochs * steps
    warmup = int(total_steps * cfg.warmup_fraction)

    player_params = [p for n, p in model.named_parameters() if p.requires_grad and n == "player_delta"]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and n != "player_delta"]
    groups = [{"params": player_params, "lr": cfg.lr_player, "base_lr": cfg.lr_player, "weight_decay": 0.0}]
    if other_params:
        groups.append({"params": other_params, "lr": cfg.lr, "base_lr": cfg.lr, "weight_decay": cfg.weight_decay})
    optimizer = torch.optim.AdamW(groups)

    # --- kaldığı yerden devam --------------------------------------------
    start_epoch, bad_epochs = 0, 0
    state_path = work_dir / STATE_FILE
    best_path = work_dir / BEST_FILE
    if store is not None:
        remote_state = store.download(STATE_FILE, work_dir)
        if remote_state is not None:
            try:
                state = torch.load(remote_state, map_location=device, weights_only=True)
                if state.get("signature") == signature and state.get("epochs") == epochs:
                    model.load_state_dict(state["model"])
                    optimizer.load_state_dict(state["optimizer"])
                    start_epoch = int(state["epoch"]) + 1
                    bad_epochs = int(state["bad_epochs"])
                    result.history = list(state["history"])
                    result.best_epoch = int(state["best_epoch"])
                    result.best_val_loss = float(state["best_val_loss"])
                    result.resumed_from_epoch = start_epoch
                    store.download(BEST_FILE, work_dir)
                    log(f"[eğitim] Hub'daki checkpoint bulundu; epoch {start_epoch + 1}'den devam ediliyor.")
                    if state.get("finished"):
                        start_epoch = epochs
                else:
                    log("[eğitim] Hub'daki checkpoint farklı veri/ayarla üretilmiş; baştan başlanıyor.")
            except Exception as exc:
                log(f"[eğitim] Checkpoint okunamadı ({exc}); baştan başlanıyor.")

    if start_epoch == 0:
        loss0, acc0 = _val_loss(model, val_data, val_rows, cfg.batch_size)
        log(f"[eğitim] başlangıç (genel model): doğrulama kaybı {loss0:.4f}, isabet %{100 * acc0:.1f}")
        result.history.append({"epoch": 0, "train_loss": None, "val_loss": loss0, "val_top1": acc0})
        result.best_epoch, result.best_val_loss = 0, loss0
        torch.save(model.state_dict(), best_path)

    def save_state(epoch: int, improved: bool, finished: bool) -> None:
        """`epoch`: 0 tabanlı, son biten epoch (devam = epoch + 1)."""
        torch.save(
            {
                "signature": signature, "epochs": epochs, "epoch": epoch, "bad_epochs": bad_epochs,
                "best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss,
                "history": result.history, "finished": finished,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            },
            state_path,
        )
        if store is not None:
            # Önce en iyi ağırlıklar, sonra durum: durum dosyası varsa en iyisi de vardır.
            if improved or epoch == 0:
                store.upload(best_path, BEST_FILE, f"en iyi ağırlıklar (epoch {result.best_epoch})")
            store.upload(state_path, STATE_FILE, f"checkpoint: epoch {epoch}")

    # --- eğitim döngüsü ---------------------------------------------------
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()
        generator = torch.Generator(device="cpu").manual_seed(cfg.seed * 1000 + epoch)
        order = train_rows_all[torch.randperm(train_rows_all.shape[0], generator=generator).to(device)]
        loss_sum, count = 0.0, 0
        for i in range(steps):
            rows = order[i * cfg.batch_size:(i + 1) * cfg.batch_size]
            if rows.shape[0] == 0:
                break
            factor = _lr_factor(epoch * steps + i, total_steps, warmup)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            tokens, se, oe, legal, target = train_data.batch(rows)
            logits, _ = model(tokens, se, oe)
            per_move = F.cross_entropy(masked_logits(logits, legal), target, reduction="none")
            w = train_data.weight[rows]
            loss = (per_move * w).sum() / w.sum().clamp_min(1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            loss_sum += loss.item() * rows.shape[0]
            count += rows.shape[0]

        val_loss, val_acc = _val_loss(model, val_data, val_rows, cfg.batch_size)
        train_loss = loss_sum / max(1, count)
        number = epoch + 1
        improved = val_loss < result.best_val_loss - cfg.min_delta
        if improved:
            result.best_epoch, result.best_val_loss, bad_epochs = number, val_loss, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
        result.history.append({
            "epoch": number, "train_loss": train_loss, "val_loss": val_loss, "val_top1": val_acc,
            "seconds": time.time() - epoch_start,
        })
        log(
            f"[eğitim] epoch {number}/{epochs}: eğitim kaybı {train_loss:.4f}, doğrulama kaybı {val_loss:.4f}, "
            f"isabet %{100 * val_acc:.1f}{' *' if improved else ''} ({format_duration(time.time() - epoch_start)})"
        )
        stop = bad_epochs >= cfg.patience
        out_of_time = (time.time() - session_start) > (cfg.time_budget_hours * 3600 - cfg.reserve_seconds)
        finished = stop or out_of_time or number == epochs
        save_state(epoch, improved, finished)
        if stop:
            result.stopped_early = True
            log(f"[eğitim] Erken durduruldu: doğrulama kaybı {cfg.patience} epoch'tur iyileşmiyor (aşırı öğrenme önlemi).")
            break
        if out_of_time:
            log("[eğitim] Süre bütçesi doldu; eğitim burada kesiliyor.")
            break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    result.seconds = time.time() - started
    if result.best_epoch == 0:
        log("[eğitim] UYARI: ince ayar doğrulama kaybını iyileştirmedi; genel model ağırlıkları korunuyor.")
    else:
        log(f"[eğitim] En iyi epoch: {result.best_epoch} (doğrulama kaybı {result.best_val_loss:.4f})")
    return result

