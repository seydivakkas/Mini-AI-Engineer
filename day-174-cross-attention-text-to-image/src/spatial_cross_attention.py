"""
Mekansal Çapraz Dikkat (Spatial Cross-Attention) Modülü (Day 174 - FAZ 9).
Görsel piksellerini (Query) CLIP/T5 metin belirteçleriyle (Key/Value) eşler ve dikkat haritası üretir.
"""

from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialCrossAttention(nn.Module):
    """Metin ve Görsel Piksellerini Eşleyen Çok Başlıklı Çapraz Dikkat Katmanı."""

    def __init__(
        self,
        query_dim: int = 128,
        context_dim: int = 256,
        heads: int = 4,
        dim_head: int = 32,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = 1.0 / math.sqrt(dim_head)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, query_dim, H, W] (Mekansal Gizli VAE Haritası)
        context: [B, S_text, context_dim] (CLIP / T5 Metin Gömmeleri)
        Döner: (out [B, query_dim, H, W], attention_map [B, H*W, S_text])
        """
        B, C, H, W = x.shape
        # [B, C, H, W] -> [B, H*W, C]
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        q = self.to_q(x_flat)  # [B, H*W, inner_dim]
        k = self.to_k(context)  # [B, S_text, inner_dim]
        v = self.to_v(context)  # [B, S_text, inner_dim]

        # Başlıklara böl: [B, heads, Tokens, dim_head]
        q = q.view(B, H * W, self.heads, -1).permute(0, 2, 1, 3)
        k = k.view(B, context.shape[1], self.heads, -1).permute(0, 2, 1, 3)
        v = v.view(B, context.shape[1], self.heads, -1).permute(0, 2, 1, 3)

        # Çapraz Dikkat Ağırlıkları: [B, heads, H*W, S_text]
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Değerlerle çarp: [B, heads, H*W, dim_head]
        out = torch.matmul(attn_weights, v)

        # Başlıkları birleştir: [B, H*W, inner_dim]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, H * W, -1)
        out = self.to_out(out)

        # [B, H*W, C] -> [B, C, H, W]
        out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Tüm başlıkların ortalama dikkat haritası: [B, H*W, S_text]
        mean_attn_map = attn_weights.mean(dim=1)

        return out, mean_attn_map
