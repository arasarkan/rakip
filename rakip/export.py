"""ONNX'e dönüştürme ve quantize etme; her sürümün benzerlik skoru karşılaştırılır.

Kural: daha küçük bir sürüm, YALNIZCA hamle eşleşme oranını düşürmüyorsa
seçilir. Sırayla int8, fp16, fp32 denenir; her birinin test setindeki eşleşme
oranı ölçülür ve tablo `meta.json` içine yazılır.

Çıktı imzası (tarayıcıdaki çıkarım kodu buna göre yazılmıştır):
  girdiler:  history  float32 [B, 64, 96]  (8 pozisyon x 12 kanal, en eski önce)
             elo_self float32 [B]
             elo_oppo float32 [B]
  çıktılar:  logits_move  float32 [B, 4352]
             logits_value float32 [B, 3]     (kayıp, beraberlik, kazanç)

Oyuncu vektörü (`player_delta`) dışa aktarımda modele gömülür.
Bu dosya torch, onnx ve onnxruntime gerektirir; yalnızca Kaggle/Colab'da çalışır.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np

from .encoding import unpack_mask
from .features import SampleSet
from .similarity import Predictions

OPSET = 17
SIZE_TARGET_BYTES = 20 * 1024 * 1024
# Quantize edilmiş sürüm, fp32'ye göre en çok bu kadar (mutlak) eşleşme kaybedebilir.
MAX_MATCH_DROP = 0.005


def export_fp32(model, path: str | Path) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, history, elo_self, elo_oppo):
            return self.inner(history, elo_self, elo_oppo, use_player=True)

    wrapper = Wrapper(model).cpu().eval()
    example = (torch.zeros(2, 64, 96), torch.full((2,), 1500.0), torch.full((2,), 1500.0))
    kwargs = dict(
        input_names=["history", "elo_self", "elo_oppo"],
        output_names=["logits_move", "logits_value"],
        dynamic_axes={name: {0: "batch"} for name in ("history", "elo_self", "elo_oppo", "logits_move", "logits_value")},
        opset_version=OPSET,
        do_constant_folding=True,
    )
    with torch.no_grad():
        try:
            torch.onnx.export(wrapper, example, str(path), dynamo=False, **kwargs)
        except TypeError:  # eski torch sürümlerinde `dynamo` parametresi yok
            torch.onnx.export(wrapper, example, str(path), **kwargs)
    return path


def quantize_int8(src: str | Path, dst: str | Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
    return Path(dst)


def convert_fp16(src: str | Path, dst: str | Path) -> Path:
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(src))
    onnx.save(float16.convert_float_to_float16(model, keep_io_types=True), str(dst))
    return Path(dst)


def onnx_predict(
    path: str | Path, samples: SampleSet, self_elo: float, rows: np.ndarray | None = None, batch_size: int = 256
) -> Predictions:
    """ONNX modeliyle (onnxruntime, CPU) tahmin; yalnızca `rows` satırları doldurulur."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    n = len(samples)
    rows = np.arange(n) if rows is None else np.asarray(rows)
    top1 = np.full(n, -1, dtype=np.int32)
    p_target = np.zeros(n, dtype=np.float32)
    p_top1 = np.zeros(n, dtype=np.float32)
    for start in range(0, rows.size, batch_size):
        r = rows[start:start + batch_size]
        feeds = {
            "history": samples.tokens(r),
            "elo_self": np.full(r.size, float(self_elo), dtype=np.float32),
            "elo_oppo": samples.oppo_elo[r].astype(np.float32),
        }
        logits = session.run(["logits_move"], feeds)[0].astype(np.float64)
        logits[~unpack_mask(samples.legal[r])] = -1e9
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        best = probs.argmax(axis=1)
        top1[r] = best
        p_top1[r] = probs[np.arange(r.size), best]
        p_target[r] = probs[np.arange(r.size), samples.target[r]]
    return Predictions(top1=top1, p_target=p_target, p_top1=p_top1)


def _match(pred: Predictions, samples: SampleSet, rows: np.ndarray) -> float:
    return float((pred.top1[rows] == samples.target[rows]).mean()) if rows.size else float("nan")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_and_choose(
    model,
    test: SampleSet,
    self_elo: float,
    out_dir: str | Path,
    torch_reference: Predictions | None = None,
    max_rows: int = 4000,
    log=print,
) -> dict:
    """fp32 / fp16 / int8 sürümlerini üretir, test setinde karşılaştırır ve en
    küçük kabul edilebilir olanı `model.onnx` olarak bırakır."""
    out_dir = Path(out_dir)
    work = out_dir / "_export"
    work.mkdir(parents=True, exist_ok=True)
    rows = np.flatnonzero(test.measured)
    if rows.size > max_rows:
        rows = np.sort(np.random.default_rng(0).choice(rows, size=max_rows, replace=False))

    variants: list[dict] = []

    def measure(name: str, path: Path) -> dict | None:
        try:
            pred = onnx_predict(path, test, self_elo, rows)
        except Exception as exc:
            log(f"[onnx] {name} çalıştırılamadı: {exc}")
            return None
        entry = {"name": name, "file": path.name, "bytes": path.stat().st_size, "match": _match(pred, test, rows)}
        if torch_reference is not None and rows.size:
            entry["agreement_with_pytorch"] = float((pred.top1[rows] == torch_reference.top1[rows]).mean())
            entry["max_prob_diff"] = float(np.abs(pred.p_top1[rows] - torch_reference.p_top1[rows]).max())
        variants.append(entry)
        log(
            f"[onnx] {name}: {entry['bytes'] / 1e6:.1f} MB, hamle eşleşmesi %{100 * entry['match']:.2f}"
            + (f", PyTorch ile aynı ilk tahmin %{100 * entry['agreement_with_pytorch']:.1f}" if "agreement_with_pytorch" in entry else "")
        )
        return entry

    fp32_path = export_fp32(model, work / "model.fp32.onnx")
    fp32 = measure("fp32", fp32_path)
    if fp32 is None:
        raise RuntimeError("ONNX dışa aktarımı çalışmıyor; fp32 model yüklenemedi.")
    if fp32.get("agreement_with_pytorch", 1.0) < 0.99:
        log("[onnx] UYARI: fp32 ONNX çıktısı PyTorch ile %99'dan az uyuşuyor; dışa aktarım hatalı olabilir.")

    for name, convert in (("int8", quantize_int8), ("fp16", convert_fp16)):
        try:
            convert(fp32_path, work / f"model.{name}.onnx")
            measure(name, work / f"model.{name}.onnx")
        except Exception as exc:
            log(f"[onnx] {name} üretilemedi: {exc}")

    baseline = fp32["match"]
    acceptable = [v for v in variants if v["match"] >= baseline - MAX_MATCH_DROP]
    chosen = min(acceptable, key=lambda v: v["bytes"])
    for v in variants:
        v["accepted"] = v in acceptable
        v["match_drop_vs_fp32"] = baseline - v["match"]
    final = out_dir / "model.onnx"
    shutil.copyfile(work / chosen["file"], final)
    shutil.rmtree(work, ignore_errors=True)

    notes = []
    rejected = [v["name"] for v in variants if not v["accepted"]]
    if rejected:
        notes.append(f"{', '.join(rejected)} sürümü benzerliği düşürdüğü için kullanılmadı.")
    if chosen["bytes"] > SIZE_TARGET_BYTES:
        notes.append(f"Model {chosen['bytes'] / 1e6:.1f} MB: 20 MB hedefinin üzerinde.")
    log(f"[onnx] seçilen sürüm: {chosen['name']} ({chosen['bytes'] / 1e6:.1f} MB)")
    for note in notes:
        log(f"[onnx] {note}")
    return {
        "chosen": chosen["name"],
        "file": "model.onnx",
        "bytes": chosen["bytes"],
        "sha256": sha256_file(final),
        "match_fp32": baseline,
        "match_chosen": chosen["match"],
        "max_allowed_drop": MAX_MATCH_DROP,
        "positions_compared": int(rows.size),
        "variants": variants,
        "notes": notes,
        "opset": OPSET,
    }
