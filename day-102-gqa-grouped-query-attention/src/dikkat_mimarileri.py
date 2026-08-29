"""
Birleşik Dikkat Mimarileri Modülü: MHA, MQA ve GQA (Day 102).
LLaMA-3, Mistral ve Gemma standartlarında Grouped-Query Attention ve KV-Repeat mekanizması.
"""

import math
from enum import Enum
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kv_cache import KVCache


class AttentionTuru(str, Enum):
    MHA = "Multi-Head Attention (MHA)"
    MQA = "Multi-Query Attention (MQA)"
    GQA = "Grouped-Query Attention (GQA)"


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Key ve Value tensörlerinin başlıklarını Query başlık sayısına eşlemek için çoğaltır (broadcast/repeat).
    Girdi:  [B, H_kv, S, D]
    Çıktı: [B, H_kv * n_rep, S, D]
    """
    if n_rep == 1:
        return x
    B, H_kv, S, D = x.shape
    # [B, H_kv, 1, S, D] -> expand -> [B, H_kv, n_rep, S, D] -> reshape -> [B, H_kv * n_rep, S, D]
    return (
        x[:, :, None, :, :]
        .expand(B, H_kv, n_rep, S, D)
        .reshape(B, H_kv * n_rep, S, D)
    )


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) Katmanı.
    Konfigürasyona bağlı olarak MHA (num_kv_heads = num_q_heads),
    MQA (num_kv_heads = 1) veya GQA (1 < num_kv_heads < num_q_heads) çalışır.
    """
    def __init__(
        self,
        dim: int = 512,
        num_q_heads: int = 8,
        num_kv_heads: int = 2,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        assert num_q_heads % num_kv_heads == 0, "Query başlık sayısı KV başlık sayısına tam bölünmelidir."
        self.num_queries_per_kv = num_q_heads // num_kv_heads
        self.head_dim = dim // num_q_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projeksiyon Matrisleri
        self.q_proj = nn.Linear(dim, num_q_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, dim, bias=bias)
        self.dropout = dropout

        # Mimari Türünün Belirlenmesi
        if num_kv_heads == num_q_heads:
            self.mimari_turu = AttentionTuru.MHA
        elif num_kv_heads == 1:
            self.mimari_turu = AttentionTuru.MQA
        else:
            self.mimari_turu = AttentionTuru.GQA

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Girdi: x -> [B, S, D]
        Çıktı: cikti -> [B, S, D], güncellenmiş kv_cache
        """
        B, S, D = x.shape

        # 1. Q, K, V Projeksiyonları
        q = self.q_proj(x).view(B, S, self.num_q_heads, self.head_dim).transpose(1, 2)  # [B, H_q, S, D_h]
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, H_kv, S, D_h]
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, H_kv, S, D_h]

        # 2. KV Cache Güncellemesi (varsa)
        if kv_cache is not None:
            k, v = kv_cache.guncelle(k, v)

        # 3. KV Başlıklarının Query Başlık Sayısına Repeat Edilmesi
        k_rep = repeat_kv(k, self.num_queries_per_kv)  # [B, H_q, S_kv, D_h]
        v_rep = repeat_kv(v, self.num_queries_per_kv)  # [B, H_q, S_kv, D_h]

        # 4. Scaled Dot-Product Attention
        drop_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k_rep, v_rep,
            dropout_p=drop_p,
            is_causal=is_causal if kv_cache is None or S > 1 else False,
        )

        # [B, H_q, S, D_h] -> [B, S, H_q * D_h] -> Out Proj
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        cikti = self.out_proj(attn_out)
        return cikti, kv_cache
