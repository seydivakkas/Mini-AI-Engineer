"""
Mistral Sliding Window Attention (SWA) Çekirdek Modülü (Day 105).
Bantlı nedensel maskeleme (Banded Causal Mask) ve Rolling Buffer Cache entegrasyonu.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rolling_buffer_cache import RollingBufferCache


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """KV başlıklarını Query başlık sayısına eşlemek için çoğaltır."""
    if n_rep == 1:
        return x
    B, H_kv, S, D = x.shape
    return x[:, :, None, :, :].expand(B, H_kv, n_rep, S, D).reshape(B, H_kv * n_rep, S, D)


def olustur_bant_maskesi(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
    """
    SWA için bantlı nedensel maske (Banded Causal Mask) üretir.
    Geçerli aralık: j <= i ve i - j < W.
    """
    i_indeks = torch.arange(seq_len, device=device).unsqueeze(1)  # [S, 1]
    j_indeks = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, S]

    # j <= i (Nedensellik) AND i - j < W (Pencere Sınırı)
    nedensel_ve_pencere = (j_indeks <= i_indeks) & ((i_indeks - j_indeks) < window_size)

    maske = torch.full((seq_len, seq_len), float("-inf"), device=device)
    maske[nedensel_ve_pencere] = 0.0
    return maske


class SlidingWindowAttention(nn.Module):
    """
    Mistral-7B Sliding Window Attention (SWA) Katmanı.
    Pencere boyutu W olan yerel dikkat uygular.
    """

    def __init__(
        self,
        dim: int = 512,
        num_q_heads: int = 8,
        num_kv_heads: int = 2,
        window_size: int = 512,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        assert num_q_heads % num_kv_heads == 0, "Query başlık sayısı KV başlık sayısına bölünmelidir."
        self.num_queries_per_kv = num_q_heads // num_kv_heads
        self.window_size = window_size
        self.head_dim = dim // num_q_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = dropout

        # Projeksiyonlar
        self.q_proj = nn.Linear(dim, num_q_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        rolling_cache: Optional[RollingBufferCache] = None,
    ) -> Tuple[torch.Tensor, Optional[RollingBufferCache]]:
        """
        Girdi: x -> [B, S, D]
        Çıktı: cikti -> [B, S, D], güncellenmiş rolling_cache
        """
        B, S, D = x.shape

        # 1. Q, K, V Projeksiyonları
        q = self.q_proj(x).view(B, S, self.num_q_heads, self.head_dim).transpose(1, 2)    # [B, H_q, S, d_h]
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, H_kv, S, d_h]
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, H_kv, S, d_h]

        # 2. Rolling Buffer Cache Güncellemesi (varsa)
        if rolling_cache is not None:
            k, v = rolling_cache.guncelle(k, v)

        S_kv = k.shape[2]

        # 3. KV Başlıklarını Query Başlık Sayısına Genişletme
        k_rep = repeat_kv(k, self.num_queries_per_kv)  # [B, H_q, S_kv, d_h]
        v_rep = repeat_kv(v, self.num_queries_per_kv)  # [B, H_q, S_kv, d_h]

        # 4. Dikkat Skoru Hesaplama
        if rolling_cache is None and S > 1:
            # Prefill Aşaması: Bant Maskeleme (Banded Mask)
            bant_maske = olustur_bant_maskesi(S, self.window_size, device=x.device)  # [S, S]
            skorlar = torch.matmul(q, k_rep.transpose(-2, -1)) * self.scale           # [B, H_q, S, S]
            skorlar = skorlar + bant_maske
            attn = F.softmax(skorlar, dim=-1)
            if self.dropout > 0.0 and self.training:
                attn = F.dropout(attn, p=self.dropout)
            attn_out = torch.matmul(attn, v_rep)
        else:
            # Decode Aşaması veya Cache Kullanımı (Cache zaten son W token'ı içerir)
            drop_p = self.dropout if self.training else 0.0
            attn_out = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                dropout_p=drop_p,
                is_causal=False if rolling_cache is not None and S == 1 else True,
            )

        # [B, H_q, S, d_h] -> [B, S, H_q * d_h] -> out_proj
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        cikti = self.out_proj(attn_out)
        return cikti, rolling_cache
