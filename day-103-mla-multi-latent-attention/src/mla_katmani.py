"""
DeepSeek Multi-Head Latent Attention (MLA) Çekirdek Modülü (Day 103).
Düşük dereceli ortak KV sıkıştırması, ayrık RoPE ve çıkarım anında matris soğurma (Matrix Absorption) içerir.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .latent_kv_cache import LatentKVCache


def uygula_rope(x: torch.Tensor, seq_len_offset: int = 0) -> torch.Tensor:
    """
    Basit ve deterministik 1D Rotary Position Embedding (RoPE) uygular.
    x: [B, H, S, D] veya [B, S, D]
    """
    is_4d = x.dim() == 4
    if not is_4d:
        # [B, S, D] -> [B, 1, S, D]
        x = x.unsqueeze(1)

    B, H, S, D = x.shape
    assert D % 2 == 0, "RoPE boyutu çift sayı olmalıdır."

    pozisyonlar = torch.arange(seq_len_offset, seq_len_offset + S, device=x.device).float()
    dim_indeksleri = torch.arange(0, D, 2, device=x.device).float()
    frekanslar = 1.0 / (10000.0 ** (dim_indeksleri / D))
    acilar = torch.outer(pozisyonlar, frekanslar)  # [S, D/2]

    sin = torch.sin(acilar)[None, None, :, :]  # [1, 1, S, D/2]
    cos = torch.cos(acilar)[None, None, :, :]  # [1, 1, S, D/2]

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    if not is_4d:
        x_rot = x_rot.squeeze(1)
    return x_rot


class MultiHeadLatentAttention(nn.Module):
    """
    DeepSeek-V2 / DeepSeek-V3 Multi-Head Latent Attention (MLA).
    - Düşük dereceli ortak KV sıkıştırması: d -> d_c
    - Düşük dereceli Q sıkıştırması: d -> d_q
    - Ayrık RoPE: d_R
    - Matris Soğurma (Matrix Absorption) optimizasyonu
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 16,
        head_dim: int = 32,
        kv_latent_dim: int = 128,
        q_latent_dim: int = 256,
        rope_dim: int = 32,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_latent_dim = kv_latent_dim
        self.q_latent_dim = q_latent_dim
        self.rope_dim = rope_dim
        self.scale = 1.0 / math.sqrt(head_dim + rope_dim)
        self.dropout = dropout

        # 1. Query Projeksiyonları (Düşük Dereceli Sıkıştırma)
        self.w_dq = nn.Linear(dim, q_latent_dim, bias=bias)
        self.w_uq = nn.Linear(q_latent_dim, num_heads * head_dim, bias=bias)
        self.w_qr = nn.Linear(q_latent_dim, num_heads * rope_dim, bias=bias)

        # 2. Key-Value Projeksiyonları (Ortak Düşük Dereceli Sıkıştırma)
        self.w_dkv = nn.Linear(dim, kv_latent_dim, bias=bias)
        self.w_uk = nn.Linear(kv_latent_dim, num_heads * head_dim, bias=bias)
        self.w_uv = nn.Linear(kv_latent_dim, num_heads * head_dim, bias=bias)
        self.w_kr = nn.Linear(dim, rope_dim, bias=bias)

        # 3. Çıktı Projeksiyonu
        self.w_out = nn.Linear(num_heads * head_dim, dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        latent_cache: Optional[LatentKVCache] = None,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[LatentKVCache]]:
        """
        MLA İleri Geçişi.
        x: [B, S, D]
        """
        B, S, D = x.shape
        offset = latent_cache.mevcut_uzunluk if latent_cache is not None else 0

        # --- 1. QUERY HESAPLAMASI ---
        c_q = self.w_dq(x)  # [B, S, d_q]
        q_c = self.w_uq(c_q).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, d_h]
        q_r = self.w_qr(c_q).view(B, S, self.num_heads, self.rope_dim).transpose(1, 2)  # [B, H, S, d_R]
        q_r = uygula_rope(q_r, seq_len_offset=offset)

        # --- 2. KEY-VALUE LATENT HESAPLAMASI ---
        c_kv = self.w_dkv(x)  # [B, S, d_c]
        k_r = self.w_kr(x)    # [B, S, d_R]
        k_r = uygula_rope(k_r, seq_len_offset=offset)

        # --- 3. ÖNBELLEK GÜNCELLEMESİ (Sadece c_kv ve k_r saklanır!) ---
        if latent_cache is not None:
            c_kv, k_r = latent_cache.guncelle(c_kv, k_r)

        S_kv = c_kv.shape[1]

        # --- 4. AÇMA (UP-PROJECTION) VE DİKKAT MATRİSİ ---
        # Key ve Value açılımları
        k_c = self.w_uk(c_kv).view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S_kv, d_h]
        v_c = self.w_uv(c_kv).view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S_kv, d_h]
        k_r_rep = k_r.unsqueeze(1).expand(B, self.num_heads, S_kv, self.rope_dim)            # [B, H, S_kv, d_R]

        # Birleşik İç Çarpım: (Q_c * K_c^T + Q_r * K_r^T) / sqrt(d_h + d_R)
        skor_icerik = torch.matmul(q_c, k_c.transpose(-2, -1))
        skor_konum = torch.matmul(q_r, k_r_rep.transpose(-2, -1))
        dikkat_skorlari = (skor_icerik + skor_konum) * self.scale

        if is_causal and (latent_cache is None or S > 1):
            maske = torch.triu(torch.full((S, S_kv), float("-inf"), device=x.device), diagonal=1 + offset)
            dikkat_skorlari = dikkat_skorlari + maske

        dikkat_olasiliklari = F.softmax(dikkat_skorlari, dim=-1)
        if self.dropout > 0.0 and self.training:
            dikkat_olasiliklari = F.dropout(dikkat_olasiliklari, p=self.dropout)

        # Çıktı bağlam vektörü: [B, H, S, d_h]
        cikti_h = torch.matmul(dikkat_olasiliklari, v_c)
        cikti_h = cikti_h.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)

        cikti = self.w_out(cikti_h)
        return cikti, latent_cache
