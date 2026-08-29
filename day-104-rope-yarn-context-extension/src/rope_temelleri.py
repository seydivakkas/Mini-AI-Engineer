"""
Rotary Position Embedding (RoPE) ve Doğrusal İnterpolasyon (PI) Modülü (Day 104).
Standart RoPE frekans hesaplamaları ve Position Interpolation ile bağlam uzatma temelleri.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class StandartRoPE(nn.Module):
    """
    Standart Rotary Position Embedding (RoPE - Su et al. 2021).
    2D alt-uzaylarda açısal rotasyon matrisi uygulayarak göreli konumu korur.
    """

    def __init__(self, dim: int = 64, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE boyutu çift sayı olmalıdır."
        self.dim = dim
        self.base = base

        # Frekanslar: theta_i = base^(-2i / dim)
        indeksler = torch.arange(0, dim, 2).float()
        self.register_buffer("frekanslar", 1.0 / (base ** (indeksler / dim)), persistent=False)

    def forward(self, x: torch.Tensor, seq_len_offset: int = 0) -> torch.Tensor:
        """
        x: [B, H, S, D] veya [B, S, D]
        """
        is_4d = x.dim() == 4
        if not is_4d:
            x = x.unsqueeze(1)

        B, H, S, D = x.shape
        pozisyonlar = torch.arange(seq_len_offset, seq_len_offset + S, device=x.device).float()
        acilar = torch.outer(pozisyonlar, self.frekanslar)  # [S, D/2]

        sin = torch.sin(acilar)[None, None, :, :]  # [1, 1, S, D/2]
        cos = torch.cos(acilar)[None, None, :, :]  # [1, 1, S, D/2]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return x_rot if is_4d else x_rot.squeeze(1)


class LinearPIRoPE(nn.Module):
    """
    Linear Position Interpolation RoPE (PI - Chen et al. 2023).
    Pozisyon indekslerini s = L_target / L_train ölçeği ile doğrusal olarak böler (m' = m / s).
    """

    def __init__(self, dim: int = 64, base: float = 10000.0, olcek: float = 4.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE boyutu çift sayı olmalıdır."
        self.dim = dim
        self.base = base
        self.olcek = max(1.0, float(olcek))

        indeksler = torch.arange(0, dim, 2).float()
        self.register_buffer("frekanslar", 1.0 / (base ** (indeksler / dim)), persistent=False)

    def forward(self, x: torch.Tensor, seq_len_offset: int = 0) -> torch.Tensor:
        is_4d = x.dim() == 4
        if not is_4d:
            x = x.unsqueeze(1)

        B, H, S, D = x.shape
        # Pozisyonları s ölçeğine bölerek interpolasyon uygula
        pozisyonlar = (torch.arange(seq_len_offset, seq_len_offset + S, device=x.device).float()) / self.olcek
        acilar = torch.outer(pozisyonlar, self.frekanslar)  # [S, D/2]

        sin = torch.sin(acilar)[None, None, :, :]
        cos = torch.cos(acilar)[None, None, :, :]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return x_rot if is_4d else x_rot.squeeze(1)
