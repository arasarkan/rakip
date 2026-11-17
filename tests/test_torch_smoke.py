"""PyTorch duman testleri: model, eğitim döngüsü, checkpoint/devam, ONNX dışa aktarım.

Yerel makinede torch kurulu olmadığı için bu testler orada ATLANIR. GitHub
Actions'ta (ücretsiz, CPU) çalışırlar; böylece eğitim kodu Kaggle'a gitmeden
önce gerçekten çalıştırılmış olur. Gerçek Maia-3 ağırlıklarını indiren test
yalnızca RAKIP_NET_TESTS=1 iken çalışır.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rakip.features import build_samples  # noqa: E402
from rakip.model import MAIA3_5M, ModelConfig, RakipModel, load_base, set_trainable  # noqa: E402
from rakip.train import TrainConfig, calibrate_elo, fit, predict, run_signature, summarize  # noqa: E402

from conftest import make_record  # noqa: E402

SMALL = ModelConfig(dim_vit=64, head_hid_dim=64, num_blocks=2, num_heads=4, dim_emb=16, gab_gen_size=16, gab_intermediate_dim=16)


def samples(start, n, plies=40):
    games = [make_record(start + i, f"2024-01-{(i % 27) + 1:02d}", color="white" if i % 2 else "black", plies=plies) for i in range(n)]
    return build_samples(games)


def test_forward_shapes_and_player_delta_starts_as_identity():
    model = RakipModel(SMALL).eval()
    data = samples(0, 2)
    tokens = torch.from_numpy(data.tokens(slice(0, 5)))
    elo = torch.full((5,), 1800.0)
    with torch.no_grad():
        with_player, value = model(tokens, elo, elo, use_player=True)
        without, _ = model(tokens, elo, elo, use_player=False)
    assert with_player.shape == (5, 4352) and value.shape == (5, 3)
    assert torch.allclose(with_player, without)          # delta sıfırken genel modelle aynı
    with torch.no_grad():
        model.player_delta.add_(0.5)
        moved, _ = model(tokens, elo, elo, use_player=True)
        still, _ = model(tokens, elo, elo, use_player=False)
    assert not torch.allclose(moved, without) and torch.allclose(still, without)


def test_promotion_logits_follow_upstream_layout():
    """Terfi girdisi = (kaynak, hedef) skoru + hedef kareye bağlı taş yanlılığı."""
    import math

    torch.manual_seed(0)
    model = RakipModel(SMALL).eval()
    tokens = torch.rand(1, 64, 96)
    elo = torch.tensor([1500.0])
    with torch.no_grad():
        logits, _ = model(tokens, elo, elo)
        x = torch.cat([tokens, model._elo_embedding(elo).unsqueeze(1).expand(-1, 64, -1),
                       model._elo_embedding(elo).unsqueeze(1).expand(-1, 64, -1)], dim=-1)
        x = model.transformer(model.token_projection(x), model.gab_shared_weight)
        sq_to = model.proj_sq_to(x)
        bias = model.promo_bias_proj(sq_to[:, 56:64]) * math.sqrt(SMALL.head_hid_dim)
    for file_from, file_to, piece in ((0, 0, 0), (3, 4, 2), (7, 6, 3)):
        index = 4096 + (file_from * 8 + file_to) * 4 + piece
        base = logits[0, (48 + file_from) * 64 + (56 + file_to)]
        assert torch.isclose(logits[0, index], base + bias[0, file_to, piece], atol=1e-5)


def test_regimes_freeze_the_right_layers():
    model = RakipModel(SMALL)
    emb = set(set_trainable(model, "embedding"))
    assert emb == {"player_delta", "proj_sq_from.weight", "proj_sq_to.weight", "promo_bias_proj.weight"}
    light = set(set_trainable(model, "light"))
    assert emb < light and any(n.startswith("transformer.layers.1.") for n in light)
    assert not any(n.startswith("token_projection") for n in light)
    full = set(set_trainable(model, "full"))
    assert light < full and not any(n.startswith("fc_value") for n in full)


def test_load_base_rejects_mismatched_weights():
    model = RakipModel(SMALL)
    state = {k: v for k, v in model.state_dict().items() if k != "player_delta"}
    state["transformer.layers.0.self_attn.gab_weight"] = model.gab_shared_weight.detach().clone()  # resmi kopya anahtar
    state["fc_ponder.weight"] = torch.zeros(1, 4)
    load_base(state, SMALL)
    broken = dict(state)
    broken.pop("proj_sq_to.weight")
    with pytest.raises(RuntimeError):
        load_base(broken, SMALL)


def test_fit_learns_resumes_and_predicts(tmp_path):
    torch.manual_seed(0)
    train, val, test = samples(0, 24), samples(100, 6), samples(200, 6)
    device = torch.device("cpu")
    model = RakipModel(SMALL)
    cfg = TrainConfig(regime="full", max_epochs=3, batch_size=32, lr=2e-3, lr_player=1e-2, patience=5).resolved(1500)
    before = summarize(predict(model, train, device, self_elo=1800), train)

    logs = []
    result = fit(model, train, val, cfg, device, 1800, run_signature({"digest": "x"}, cfg, 1800), tmp_path, log=logs.append)
    assert any("Tahmini eğitim süresi" in line for line in logs)
    assert len(result.history) == 4 and result.history[0]["epoch"] == 0
    assert result.history[-1]["train_loss"] < result.history[1]["train_loss"]     # öğreniyor
    assert (tmp_path / "checkpoints" / "state.pt").exists() and (tmp_path / "checkpoints" / "best.pt").exists()

    pred = predict(model, test, device, self_elo=1800, seed=3)
    assert pred.top1.shape == (len(test),) and pred.sample.shape == (len(test),)
    legal = test.legal_mask(slice(None))
    rows = np.arange(len(test))
    assert legal[rows, pred.top1].all() and legal[rows, pred.sample].all()        # yalnızca yasal hamleler
    assert np.all((pred.p_target >= 0) & (pred.p_target <= pred.p_top1 + 1e-6))
    assert summarize(predict(model, train, device, self_elo=1800), train)["nll"] < before["nll"]

    best, table = calibrate_elo(model, val, device, 1800, log=lambda *_: None)
    assert best in [row["elo"] for row in table] and len(table) == 7


def test_early_stopping_keeps_best_weights(tmp_path):
    torch.manual_seed(1)
    train, val = samples(0, 8), samples(300, 6)
    model = RakipModel(SMALL)
    cfg = TrainConfig(regime="full", max_epochs=12, batch_size=16, lr=2e-2, lr_player=2e-2, patience=1).resolved(1500)
    result = fit(model, train, val, cfg, torch.device("cpu"), 1800, "sig", tmp_path, log=lambda *_: None)
    if result.stopped_early:
        assert len(result.history) < 13
    best = min(h["val_loss"] for h in result.history)
    assert result.best_val_loss == pytest.approx(best)


class MemoryStore:
    """Hub yerine bellek içi depo: checkpoint yükleme ve devam etmeyi sınar."""

    def __init__(self):
        self.files = {}

    def upload(self, local, remote, message=None, retries=3):
        with open(local, "rb") as fh:
            self.files[remote] = fh.read()
        return True

    def download(self, remote, local_dir):
        if remote not in self.files:
            return None
        from pathlib import Path

        path = Path(local_dir) / remote
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[remote])
        return path


def test_resume_from_hub_checkpoint_after_interruption(tmp_path):
    torch.manual_seed(2)
    train, val = samples(0, 16), samples(100, 6)
    device = torch.device("cpu")
    store = MemoryStore()
    cfg = TrainConfig(regime="full", max_epochs=3, batch_size=32, lr=1e-3, patience=9).resolved(1500)

    # 1. seans: bir epoch sonra "kopuyor" (zaman bütçesi dolmuş gibi).
    first_cfg = TrainConfig(**{**cfg.__dict__})
    model = RakipModel(SMALL)
    initial = {k: v.clone() for k, v in model.state_dict().items()}

    class Interrupt(Exception):
        pass

    calls = {"n": 0}
    original_upload = store.upload

    def upload_then_die(local, remote, message=None, retries=3):
        original_upload(local, remote, message, retries)
        if remote.endswith("state.pt"):
            calls["n"] += 1
            if calls["n"] == 1:
                raise Interrupt()
        return True

    store.upload = upload_then_die
    with pytest.raises(Interrupt):
        fit(model, train, val, first_cfg, device, 1800, "sig", tmp_path / "a", store=store, log=lambda *_: None)
    assert "checkpoints/state.pt" in store.files and "checkpoints/best.pt" in store.files

    # 2. seans: taze model, Hub'daki checkpoint'ten devam etmeli.
    store.upload = original_upload
    fresh = RakipModel(SMALL)
    fresh.load_state_dict(initial)
    logs = []
    result = fit(fresh, train, val, cfg, device, 1800, "sig", tmp_path / "b", store=store, log=logs.append)
    assert result.resumed_from_epoch == 1
    assert [h["epoch"] for h in result.history] == [0, 1, 2, 3]
    assert any("devam ediliyor" in line for line in logs)

    # Farklı veri/ayar imzasıyla eski checkpoint kullanılmaz.
    other = RakipModel(SMALL)
    result2 = fit(other, train, val, cfg, device, 1800, "baska-imza", tmp_path / "c", store=store, log=lambda *_: None)
    assert result2.resumed_from_epoch is None


def test_onnx_export_matches_pytorch_and_reports_variants(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from rakip.export import export_and_choose, onnx_predict

    torch.manual_seed(3)
    model = RakipModel(SMALL).eval()
    with torch.no_grad():
        model.player_delta.normal_(0, 0.3)      # oyuncu vektörü dışa aktarıma gömülmeli
    test = samples(400, 6)
    device = torch.device("cpu")
    reference = predict(model, test, device, self_elo=1800)
    info = export_and_choose(model, test, 1800, tmp_path, torch_reference=reference, log=lambda *_: None)
    assert (tmp_path / "model.onnx").exists() and info["bytes"] == (tmp_path / "model.onnx").stat().st_size
    fp32 = next(v for v in info["variants"] if v["name"] == "fp32")
    assert fp32["agreement_with_pytorch"] >= 0.99 and fp32["max_prob_diff"] < 1e-3
    chosen = next(v for v in info["variants"] if v["name"] == info["chosen"])
    assert chosen["accepted"] and chosen["match"] >= fp32["match"] - info["max_allowed_drop"]
    # Değişken toplu iş boyutu ve imza
    import onnxruntime as ort

    session = ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == ["history", "elo_self", "elo_oppo"]
    assert [o.name for o in session.get_outputs()] == ["logits_move", "logits_value"]
    rows = np.flatnonzero(test.measured)[:7]
    pred = onnx_predict(tmp_path / "model.onnx", test, 1800, rows)
    assert (pred.top1[rows] >= 0).all()


@pytest.mark.skipif(os.environ.get("RAKIP_NET_TESTS") != "1", reason="ağ gerektirir (RAKIP_NET_TESTS=1)")
def test_real_maia3_weights_load_and_play_like_a_human():
    """Yeniden yazılan mimari, resmi Maia3-5M ağırlıklarıyla anlamlı tahmin üretmeli."""
    import chess

    from rakip import encoding as enc
    from rakip.train import load_base_from_hub

    model = load_base_from_hub(device=torch.device("cpu"))
    assert sum(p.numel() for p in model.parameters()) > 5_000_000
    board = chess.Board()
    codes = np.stack([enc.board_codes(board)])
    tokens = torch.from_numpy(enc.stack_history(codes, enc.history_indices(0)[None, :]))
    elo = torch.tensor([1900.0])
    with torch.no_grad():
        logits, _ = model(tokens, elo, elo, use_player=False)
    mask = torch.from_numpy(enc.legal_mask(board))
    probs = torch.softmax(logits[0].masked_fill(~mask, -1e9), dim=-1)
    top = [enc.index_move(int(i), board.turn).uci() for i in torch.topk(probs, 2).indices]
    # Tarayıcıdaki resmi ONNX sürümü aynı pozisyonda e4 ~%59, d4 ~%27 veriyor.
    assert top == ["e2e4", "d2d4"]
    assert 0.50 < float(probs.max()) < 0.68
    assert MAIA3_5M.dim_vit == 256


def test_fit_with_weighted_extra_games(tmp_path):
    """Ek tempo oyunları düşük ağırlıkla eğitime girer; ağırlık 0 olan örnekler kaybı etkilemez."""
    torch.manual_seed(4)
    games = [make_record(i, f"2024-01-{(i % 27) + 1:02d}", plies=30) for i in range(20)]
    extra_ids = {g.game_id for g in games[:10]}
    weighted = build_samples(games, game_weight=lambda g: 0.25 if g.game_id in extra_ids else 1.0)
    val = samples(500, 4)
    model = RakipModel(SMALL)
    cfg = TrainConfig(regime="full", max_epochs=2, batch_size=32, patience=5).resolved(1500)
    result = fit(model, weighted, val, cfg, torch.device("cpu"), 1800, "w", tmp_path, log=lambda *_: None)
    assert len(result.history) == 3 and all(np.isfinite(h["val_loss"]) for h in result.history)
    zero = build_samples(games, game_weight=lambda g: 0.0 if g.game_id in extra_ids else 1.0)
    result2 = fit(RakipModel(SMALL), zero, val, cfg, torch.device("cpu"), 1800, "z", tmp_path / "z", log=lambda *_: None)
    assert all(np.isfinite(h["val_loss"]) for h in result2.history)
