"""
Zaman Koşullu Denoising UNet Modülü (Day 172 - FAZ 9).
Sinüzoidal Zaman Gömüşü (Sinusoidal Time Embedding) ile VAE Gizli Uzayında Gürültü Kestirimi.
"""

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinuzoidalZamanGomusu(nn.Module):
    """Transformer ve Difüzyon Modelleri İçin Sinüzoidal Zaman Kodlayıcı."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """timesteps: [B] -> [B, dim]"""
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class DenoisingUNet(nn.Module):
    """VAE Gizli Uzayındaki (4x64x64) Gürültüyü Kestiren UNet Mimarisi."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4, base_channels: int = 64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinuzoidalZamanGomusu(base_channels),
            nn.Linear(base_channels, base_channels * 4),
            nn.GELU(),
            nn.Linear(base_channels * 4, base_channels * 4),
        )

        # Down-blocks (Aşağı Örnekleme)
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)

        # Bottleneck (Darboğaz)
        self.mid_conv = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1)

        # Up-blocks (Yukarı Örnekleme) + Skip Connections
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels, kernel_size=4, stride=2, padding=1)
        self.conv_out = nn.Conv2d(base_channels * 2, out_channels, kernel_size=3, padding=1)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z_t: [B, in_channels=4, H=16, W=16]
        t: [B]
        Döner: epsilon_theta [B, out_channels=4, H=16, W=16]
        """
        t_emb = self.time_mlp(t)  # [B, base_channels*4]

        # Down 1
        x1 = F.gelu(self.conv_in(z_t))
        x2 = F.gelu(self.down1(x1))
        x3 = F.gelu(self.down2(x2))

        # Bottleneck + Zaman Koşullandırma
        t_proj = t_emb.view(t_emb.shape[0], t_emb.shape[1], 1, 1)
        mid = F.gelu(self.mid_conv(x3) + t_proj)

        # Up 1
        u1 = F.gelu(self.up1(mid))
        u1_cat = torch.cat([u1, x2], dim=1)  # Skip connection

        # Up 2
        u2 = F.gelu(self.up2(u1_cat))
        u2_cat = torch.cat([u2, x1], dim=1)  # Skip connection

        out = self.conv_out(u2_cat)
        return out
