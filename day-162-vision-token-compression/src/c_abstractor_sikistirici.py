"""
C-Abstractor (Convolutional Abstractor) Token Sıkıştırma Modülü (Day 162 - FAZ 9).
2D Derinlikli Konvolüsyon (Depthwise Conv) ile 256 tokenı 64 tokena (%75 sıkıştırma) indirger.
"""

import math
import torch
import torch.nn as nn


class CAbstractorSikistirici(nn.Module):
    """LLaVA-NeXT ve Honeybee tarzı C-Abstractor Token Sıkıştırıcı."""

    def __init__(self, d_vision: int = 768, d_model: int = 512, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.in_proj = nn.Linear(d_vision, d_model)

        # 2D Derinlikli Konvolüsyonel Downsampling Blokları
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=stride, padding=1, groups=d_model),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1, stride=1),
            nn.GELU(),
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Girdi: (Batch, 256, 768) -> 16x16 ızgara
        Çıktı: (Batch, 64, 512) -> Stride=2 ile 8x8 ızgara (64 token)
        """
        B, N, C = visual_tokens.shape
        H = W = int(math.sqrt(N))  # 16

        # Lineer Projeksiyon: (B, 256, 512)
        x = self.in_proj(visual_tokens)

        # 2D Grid'e Dönüştür: (B, 512, 16, 16)
        x_2d = x.transpose(1, 2).view(B, -1, H, W)

        # Konvolüsyonel İndirgeme: (B, 512, 8, 8)
        x_down = self.conv(x_2d)

        # Diziye Düzleştir: (B, 64, 512)
        x_flat = x_down.flatten(2).transpose(1, 2)
        out = self.out_proj(x_flat)
        return out
