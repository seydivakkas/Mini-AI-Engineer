"""
Spatial Pooling (2x2 / 4x4) Token Sıkıştırma Modülü (Day 162 - FAZ 9).
Adaptif 2D Ortalama ve Maksimum Havuzlama ile token sayısını 4x-16x azaltır.
"""

import math
import torch
import torch.nn as nn


class SpatialPoolingSikistirici(nn.Module):
    """Adaptif Spatial Pooling (2x2 veya 4x4) Token Sıkıştırıcı."""

    def __init__(self, d_vision: int = 768, d_model: int = 512, pool_boyutu: int = 2):
        super().__init__()
        self.pool_boyutu = pool_boyutu
        self.in_proj = nn.Linear(d_vision, d_model)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(16 // pool_boyutu, 16 // pool_boyutu))

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Girdi: (Batch, 256, 768)
        Çıktı: (Batch, 64, 512) [pool_boyutu=2 için]
        """
        B, N, C = visual_tokens.shape
        H = W = int(math.sqrt(N))  # 16

        x = self.in_proj(visual_tokens)
        x_2d = x.transpose(1, 2).view(B, -1, H, W)  # (B, 512, 16, 16)
        x_pooled = self.pool(x_2d)                  # (B, 512, 8, 8)
        out = x_pooled.flatten(2).transpose(1, 2)   # (B, 64, 512)
        return out
