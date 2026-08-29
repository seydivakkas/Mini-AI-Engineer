"""
NTK-Aware Scaling ve YaRN (Yet another RoPE extensioN) Modülü (Day 104).
128k+ bağlam uzatmalarında yüksek/düşük frekans ayrımı ve dikkat entropi düzeltmesi.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class NTKAwareRoPE(nn.Module):
    """
    NTK-Aware RoPE Scaling (bloc97).
    Pozisyonlar yerine RoPE taban frekansını (base) ölçekler:
    base' = base * s^(d / (d-2))
    Bu sayede yüksek frekanslı yerel detaylar bozulmadan uzun menzilli düşük frekanslar genişletilir.
    """

    def __init__(self, dim: int = 64, base: float = 10000.0, olcek: float = 4.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE boyutu çift sayı olmalıdır."
        self.dim = dim
        self.olcek = max(1.0, float(olcek))

        # NTK Taban Frekans Dönüşümü
        ntk_us = dim / (dim - 2)
        self.ntk_base = base * (self.olcek ** ntk_us)

        indeksler = torch.arange(0, dim, 2).float()
        self.register_buffer("frekanslar", 1.0 / (self.ntk_base ** (indeksler / dim)), persistent=False)

    def forward(self, x: torch.Tensor, seq_len_offset: int = 0) -> torch.Tensor:
        is_4d = x.dim() == 4
        if not is_4d:
            x = x.unsqueeze(1)

        B, H, S, D = x.shape
        pozisyonlar = torch.arange(seq_len_offset, seq_len_offset + S, device=x.device).float()
        acilar = torch.outer(pozisyonlar, self.frekanslar)

        sin = torch.sin(acilar)[None, None, :, :]
        cos = torch.cos(acilar)[None, None, :, :]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return x_rot if is_4d else x_rot.squeeze(1)


class YaRNRoPE(nn.Module):
    """
    YaRN (Yet another RoPE extensioN - Peng et al. 2023).
    - Dalga boyu (Wavelength) bazlı rampa interpolasyonu: gamma(t)
    - Yüksek frekanslar: Tam ekstrapolasyon (sıfır bozulma)
    - Düşük frekanslar: Tam interpolasyon
    - Orta frekanslar: Yumuşak rampa geçişi
    - Dikkat Entropi Sıcaklık Çarpanı (Attention Temperature Scaling)
    """

    def __init__(
        self,
        dim: int = 64,
        base: float = 10000.0,
        olcek: float = 4.0,
        orijinal_max_seq_len: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        super().__init__()
        assert dim % 2 == 0, "RoPE boyutu çift sayı olmalıdır."
        self.dim = dim
        self.base = base
        self.olcek = max(1.0, float(olcek))
        self.orijinal_max_seq_len = orijinal_max_seq_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # 1. YaRN Hibrit Frekanslarının Hesaplanması
        indeksler = torch.arange(0, dim, 2).float()
        orijinal_frekanslar = 1.0 / (base ** (indeksler / dim))

        # Dalga boyu: lambda_i = 2 * pi / theta_i
        dalga_boylari = 2.0 * math.pi / orijinal_frekanslar

        # Rampa katsayıları: gamma_i
        # r_i = L_train / lambda_i
        r = orijinal_max_seq_len / dalga_boylari
        gamma = torch.clamp((r - beta_slow) / (beta_fast - beta_slow), min=0.0, max=1.0)

        # Hibrit İnterpolasyon: theta_yarn = (1 - gamma) * (theta / s) + gamma * theta
        yarn_frekanslar = (1.0 - gamma) * (orijinal_frekanslar / self.olcek) + gamma * orijinal_frekanslar
        self.register_buffer("frekanslar", yarn_frekanslar, persistent=False)

        # 2. Dikkat Sıcaklık Ölçeği (Temperature Factor)
        # t = 0.1 * ln(s) + 1.0
        self.sicaklik_katsayisi = 0.1 * math.log(self.olcek) + 1.0 if self.olcek > 1.0 else 1.0

    def forward(self, x: torch.Tensor, seq_len_offset: int = 0) -> torch.Tensor:
        is_4d = x.dim() == 4
        if not is_4d:
            x = x.unsqueeze(1)

        B, H, S, D = x.shape
        pozisyonlar = torch.arange(seq_len_offset, seq_len_offset + S, device=x.device).float()
        acilar = torch.outer(pozisyonlar, self.frekanslar)

        sin = torch.sin(acilar)[None, None, :, :]
        cos = torch.cos(acilar)[None, None, :, :]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return x_rot if is_4d else x_rot.squeeze(1)
