"""
Diffusion Transformer (DiT) Bloğu Modülü (Day 177 - FAZ 9).
Multi-Head Self-Attention, Feed-Forward Network ve adaLN-Zero modülasyonu içerir.
"""

import torch
import torch.nn as nn
from .adaln_zero import AdaLNZero, modulate


class DiTBlock(nn.Module):
    """
    adaLN-Zero ile modüle edilmiş Diffusion Transformer Bloğu.
    """

    def __init__(self, hidden_size: int, num_heads: int, cond_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        self.adaln_zero = AdaLNZero(hidden_size, cond_size)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D] (Yama tokenları dizisi)
        c: [B, cond_size] (Zaman adımı t ve Metin/Sınıf gömmesi)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_zero(c)

        # 1. Attention Alt Bloğu
        norm_x1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(norm_x1, norm_x1, norm_x1)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # 2. MLP Alt Bloğu
        norm_x2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(norm_x2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x
