"""Maia-3 (Chessformer) modelinin ince ayara ve ONNX'e uygun yeniden yazımı.

Resmi kod: https://github.com/CSSLab/maia3 (AGPL-3.0). Parametre adları resmi
`state_dict` ile aynıdır; `load_base` resmi ağırlıkları doğrudan yükler.

Resmi koddan farklar (davranış aynı):
- Dikkat katmanı `nn.MultiheadAttention` yerine açık matris çarpımıyla yazıldı.
  Böylece ONNX'e temiz aktarılır ve değişken toplu iş boyutunda çalışır.
- RMSNorm elle yazıldı (eski torch sürümleri ve ONNX için).
- `player_delta`: oyuncuya özel, sıfırdan başlayan 128 boyutlu bir vektör.
  Kendi reyting embedding'ine eklenir. Sıfırken model, genel Maia-3 ile
  birebir aynıdır; ince ayar bu vektörü "bu oyuncu" yönünde kaydırır.
- Düşünme süresi (ponder) başlığı yok; kullanılmıyor.

Bu dosya torch gerektirir ve yalnızca Kaggle/Colab'da çalışır.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import HISTORY, N_MOVES


@dataclass(frozen=True)
class ModelConfig:
    history: int = HISTORY
    dim_emb: int = 128
    dim_vit: int = 256
    head_hid_dim: int = 256
    num_blocks: int = 8
    num_heads: int = 8
    mlp_ratio: float = 2.0
    gab_gen_size: int = 64
    gab_intermediate_dim: int = 64
    elo_upper: float = 5000.0

    def to_dict(self) -> dict:
        return asdict(self)


MAIA3_5M = ModelConfig()


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1.1920929e-07  # torch.nn.RMSNorm varsayılanı (float32 eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class _AttentionParams(nn.Module):
    """nn.MultiheadAttention(bias=False) ile aynı parametre adları."""

    def __init__(self, dim: int):
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.out_proj = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_uniform_(self.in_proj_weight)


class GabAttention(nn.Module):
    """Geometrik Dikkat Yanlılığı (GAB) ile çok başlı öz-dikkat."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, h = cfg.dim_vit, cfg.num_heads
        self.num_heads = h
        self.head_dim = d // h
        self.gen_size = cfg.gab_gen_size
        self.mha = _AttentionParams(d)
        self.sm2 = nn.Linear(d, cfg.gab_intermediate_dim)
        self.ln1 = nn.LayerNorm(cfg.gab_intermediate_dim)
        self.sm3 = nn.Linear(cfg.gab_intermediate_dim, h * cfg.gab_gen_size)
        self.ln2 = nn.LayerNorm(h * cfg.gab_gen_size)

    def forward(self, x: torch.Tensor, gab_weight: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        # Pozisyona bağlı kare-çifti yanlılığı: (B, H, 64, 64)
        y = F.gelu(self.sm2(x.mean(dim=1)))
        y = self.ln1(y)
        y = F.gelu(self.sm3(y))
        y = self.ln2(y).view(b, self.num_heads, self.gen_size)
        bias = torch.matmul(y, gab_weight.t()).view(b, self.num_heads, 64, 64)

        qkv = F.linear(x, self.mha.in_proj_weight)                     # (B, 64, 3d)
        qkv = qkv.view(b, 64, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                               # (B, H, 64, dh)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim) + bias
        out = torch.matmul(torch.softmax(scores, dim=-1), v)           # (B, H, 64, dh)
        out = out.transpose(1, 2).reshape(b, 64, self.num_heads * self.head_dim)
        return self.mha.out_proj(out)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.dim_vit
        self.self_attn = GabAttention(cfg)
        self.linear1 = nn.Linear(d, int(d * cfg.mlp_ratio))
        self.linear2 = nn.Linear(int(d * cfg.mlp_ratio), d)
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)

    def forward(self, x: torch.Tensor, gab_weight: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.self_attn(x, gab_weight))
        x = self.norm2(x + self.linear2(F.gelu(self.linear1(x))))
        return x


class _Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.num_blocks)])
        self.norm = nn.LayerNorm(cfg.dim_vit)

    def forward(self, x: torch.Tensor, gab_weight: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, gab_weight)
        return self.norm(x)


class RakipModel(nn.Module):
    def __init__(self, cfg: ModelConfig = MAIA3_5M):
        super().__init__()
        self.cfg = cfg
        self.elo_embedding_low = nn.Embedding(1, cfg.dim_emb)
        self.elo_embedding_high = nn.Embedding(1, cfg.dim_emb)
        self.token_projection = nn.Linear(12 * cfg.history + 2 * cfg.dim_emb, cfg.dim_vit)
        self.gab_shared_weight = nn.Parameter(torch.empty(64 * 64, cfg.gab_gen_size))
        nn.init.xavier_normal_(self.gab_shared_weight)
        self.transformer = _Encoder(cfg)
        self.last_ln = nn.LayerNorm(cfg.dim_vit)
        self.fc_value_hid = nn.Linear(cfg.dim_vit, cfg.head_hid_dim)
        self.fc_value = nn.Linear(cfg.head_hid_dim, 3)
        self.proj_sq_from = nn.Linear(cfg.dim_vit, cfg.head_hid_dim, bias=False)
        self.proj_sq_to = nn.Linear(cfg.dim_vit, cfg.head_hid_dim, bias=False)
        self.promo_bias_proj = nn.Linear(cfg.head_hid_dim, 4, bias=False)
        # Oyuncuya özel kayma; sıfır = genel model.
        self.player_delta = nn.Parameter(torch.zeros(cfg.dim_emb))

    def _elo_embedding(self, elos: torch.Tensor) -> torch.Tensor:
        # Resmi koddaki gibi: iki uç embedding arasında doğrusal ara değer.
        w_low = (elos.float().clamp(0, self.cfg.elo_upper) / self.cfg.elo_upper).unsqueeze(1)
        return w_low * self.elo_embedding_low.weight + (1.0 - w_low) * self.elo_embedding_high.weight

    def forward(
        self,
        tokens: torch.Tensor,      # (B, 64, 96)
        self_elos: torch.Tensor,   # (B,)
        oppo_elos: torch.Tensor,   # (B,)
        use_player: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = tokens.shape[0]
        self_emb = self._elo_embedding(self_elos)
        if use_player:
            self_emb = self_emb + self.player_delta
        oppo_emb = self._elo_embedding(oppo_elos)
        x = torch.cat(
            [tokens, self_emb.unsqueeze(1).expand(-1, 64, -1), oppo_emb.unsqueeze(1).expand(-1, 64, -1)], dim=-1
        )
        x = self.token_projection(x)
        x = self.transformer(x, self.gab_shared_weight)

        d = self.cfg.head_hid_dim
        sq_from = self.proj_sq_from(x)
        sq_to = self.proj_sq_to(x)
        scores = torch.matmul(sq_from, sq_to.transpose(1, 2)) / math.sqrt(d)   # (B, 64, 64) [kaynak, hedef]
        # Terfiler: 7. yatay (48-55) -> 8. yatay (56-63), hedef kareye bağlı taş yanlılığı.
        promo_bias = self.promo_bias_proj(sq_to[:, 56:64, :]) * math.sqrt(d)   # (B, 8, 4)
        promo = scores[:, 48:56, 56:64].unsqueeze(-1) + promo_bias.unsqueeze(1)  # (B, 8, 8, 4)
        logits_move = torch.cat([scores.reshape(b, 64 * 64), promo.reshape(b, 256)], dim=1)

        pooled = self.last_ln(x.mean(dim=1))
        logits_value = self.fc_value(F.relu(self.fc_value_hid(pooled)))
        return logits_move, logits_value


def load_base(state_dict: dict, cfg: ModelConfig = MAIA3_5M) -> RakipModel:
    """Resmi Maia-3 `state_dict`'ini yükler; eksik ya da fazla anahtar varsa durur."""
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("smolgen", "gab")
        if key.startswith("module."):
            key = key[len("module."):]
        if key.endswith(".self_attn.gab_weight"):
            continue  # paylaşılan gab_shared_weight'in katman başına kopyası
        if key.startswith(("fc_ponder", "fc_ponder_hid")):
            continue
        cleaned[key] = value
    model = RakipModel(cfg)
    result = model.load_state_dict(cleaned, strict=False)
    missing = [k for k in result.missing_keys if k != "player_delta"]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            "Temel model ağırlıkları mimariyle uyuşmuyor.\n"
            f"  eksik: {missing[:8]}\n  fazla: {list(result.unexpected_keys)[:8]}"
        )
    assert logits_shape_ok(model), "çıktı boyutu beklenen hamle sözlüğüyle uyuşmuyor"
    return model


def logits_shape_ok(model: RakipModel) -> bool:
    model.eval()
    with torch.no_grad():
        tokens = torch.zeros(2, 64, 12 * model.cfg.history)
        elos = torch.tensor([1500.0, 2500.0])
        logits, value = model(tokens, elos, elos)
    return tuple(logits.shape) == (2, N_MOVES) and tuple(value.shape) == (2, 3)


REGIMES = ("embedding", "light", "full")


def set_trainable(model: RakipModel, regime: str) -> list[str]:
    """Eğitim rejimine göre katmanları dondurur; eğitilen parametre adlarını döner.

    - embedding: yalnızca oyuncu vektörü + hamle başlığı (çok az veri)
    - light:     + son iki dönüştürücü blok (1000 oyundan az)
    - full:      değer başlığı dışında her şey
    """
    if regime not in REGIMES:
        raise ValueError(f"rejim şunlardan biri olmalı: {REGIMES}")
    head = ("player_delta", "proj_sq_from", "proj_sq_to", "promo_bias_proj")
    last = model.cfg.num_blocks
    light = head + tuple(f"transformer.layers.{i}." for i in (last - 2, last - 1)) + ("transformer.norm",)
    value = ("fc_value", "fc_value_hid", "last_ln")
    names = []
    for name, param in model.named_parameters():
        if regime == "embedding":
            train = name.startswith(head)
        elif regime == "light":
            train = name.startswith(light)
        else:
            train = not name.startswith(value)
        param.requires_grad_(train)
        if train:
            names.append(name)
    return names
