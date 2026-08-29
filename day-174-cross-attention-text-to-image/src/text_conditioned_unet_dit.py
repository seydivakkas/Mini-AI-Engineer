"""
Metin Koşullu Difüzyon Bloğu (UNet / DiT Transformer Bloğu) Modülü (Day 174 - FAZ 9).
Residual Conv + Spatial Self-Attention + Spatial Cross-Attention katmanlarını birleştirir.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .spatial_cross_attention import SpatialCrossAttention


class TextConditionedDiffusionBlock(nn.Module):
    """Stable Diffusion ve Diffusion Transformer (DiT) Hibrit Bloğu."""

    def __init__(
        self,
        channels: int = 128,
        context_dim: int = 256,
        heads: int = 4,
    ):
        super().__init__()
        # 1. Konvolüsyonel Özellik İşleme
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)

        # 2. Mekansal Öz-Dikkat (Self-Attention: Piksel-Piksel İlişkisi)
        self.self_attn = SpatialCrossAttention(query_dim=channels, context_dim=channels, heads=heads)
        self.norm2 = nn.GroupNorm(8, channels)

        # 3. Mekansal Çapraz-Dikkat (Cross-Attention: Piksel-Metin İlişkisi)
        self.cross_attn = SpatialCrossAttention(query_dim=channels, context_dim=context_dim, heads=heads)
        self.norm3 = nn.GroupNorm(8, channels)

        # 4. İleri Besleme (FFN)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
        )
        self.norm4 = nn.GroupNorm(8, channels)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, channels, H, W]
        context: [B, S_text, context_dim]
        Döner: (out [B, channels, H, W], cross_attn_map [B, H*W, S_text])
        """
        # Adım 1: Conv + Norm
        h = F.gelu(self.norm1(self.conv1(x))) + x

        # Adım 2: Self-Attention
        h_norm = self.norm2(h)
        B, C, H, W = h_norm.shape
        self_context = h_norm.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        self_out, _ = self.self_attn(h_norm, self_context)
        h = h + self_out

        # Adım 3: Cross-Attention
        h_norm = self.norm3(h)
        cross_out, attn_map = self.cross_attn(h_norm, context)
        h = h + cross_out

        # Adım 4: FFN + Norm
        h = h + self.ffn(self.norm4(h))

        return h, attn_map
