"""
Uzamsal-Zamansal (Spatio-Temporal) 3D Dikkat Modülü (Day 167 - FAZ 9).
Ayrıştırılmış (Factorized) Uzay-Zaman Dikkati (Spatial Attention + Temporal Attention).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalAttention(nn.Module):
    """
    Video patch tokenları için Space-Time Factorized Self-Attention.
    Girdi: [Batch, Zaman (T), Uzay (N), D_model]
    """

    def __init__(self, d_model: int = 256, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # 1. Uzamsal Dikkat (Spatial Attention)
        self.spatial_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm_spatial = nn.LayerNorm(d_model)

        # 2. Zamansal Dikkat (Temporal Attention)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm_temporal = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, N, D]
        """
        B, T, N, D = x.shape

        # A. UZAMSAL DİKKAT (Her kare kendi içinde N token arasında dikkat kurar)
        # [B*T, N, D]
        x_space = x.view(B * T, N, D)
        norm_space = self.norm_spatial(x_space)
        out_space, _ = self.spatial_attn(norm_space, norm_space, norm_space)
        x_space = x_space + out_space

        # Geriye [B, T, N, D] yapısına dönüştür
        x = x_space.view(B, T, N, D)

        # B. ZAMANSAL DİKKAT (Aynı uzamsal pozisyondaki patch'ler T zaman adımı boyunca dikkat kurar)
        # [B, N, T, D] -> [B*N, T, D]
        x_time = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
        norm_time = self.norm_temporal(x_time)
        out_time, _ = self.temporal_attn(norm_time, norm_time, norm_time)
        x_time = x_time + out_time

        # Geriye [B, T, N, D] yapısına dönüştür
        out = x_time.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()
        return out
