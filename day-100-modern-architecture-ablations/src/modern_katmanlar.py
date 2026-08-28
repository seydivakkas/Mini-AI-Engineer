"""
Modern Transformer Katmanları Modülü (Day 100).
SwiGLU, RMSNorm ve Scaled Dot-Product Attention (SDPA / FlashAttention) bileşenleri.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .konfigurasyon import ModernMiniViTConfig


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).
    LayerNorm'a kıyasla ortalama çıkarma maliyetini ortadan kaldırarak %7-15 hızlanma sağlar.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit (SwiGLU).
    Modern LLM'lerin (LLaMA, Mistral) kullandığı kapılı aktivasyon bloğu.
    """
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0):
        super().__init__()
        # SwiGLU parametre dengesi için genellikle hidden_features = int(2/3 * 4d) alınır
        self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
        self.w_up = nn.Linear(in_features, hidden_features, bias=False)
        self.w_down = nn.Linear(hidden_features, in_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        out = self.w_down(gate * up)
        return self.dropout(out)


class GELUFFN(nn.Module):
    """Geleneksel 2 Katmanlı GELU İleri Besleme Bloğu (Baseline)."""
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class ModernDikkatSDPA(nn.Module):
    """
    PyTorch 2.0 Scaled Dot-Product Attention (SDPA / FlashAttention) Çoklu Başlıklı Dikkat.
    Bellek ayak izini O(N^2)'den O(N)'ye düşürür ve donanım çekirdek optimizasyonu sağlar.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "Boyut başlık sayısına tam bölünmelidir."
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        B, N, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # [3, B, num_heads, N, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # PyTorch SDPA (FlashAttention / Mem-Efficient backend)
        drop_rate = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=drop_rate,
            is_causal=False,
        )

        # [B, num_heads, N, head_dim] -> [B, N, D]
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class ModernTransformerBlok(nn.Module):
    """
    Ablasyon yapılabilir Modern Transformer Bloğu.
    Norm (RMSNorm/LayerNorm), Dikkat (SDPA/Standart) ve FFN (SwiGLU/GELU) kombinasyonlarını destekler.
    """
    def __init__(self, config: ModernMiniViTConfig):
        super().__init__()
        self.config = config

        # 1. Normalizasyon 1
        if config.norm_turu == "rmsnorm":
            self.norm1 = RMSNorm(config.gizli_boyut)
        else:
            self.norm1 = nn.LayerNorm(config.gizli_boyut)

        # 2. Dikkat Katmanı
        if config.dikkat_turu == "sdpa":
            self.dikkat = ModernDikkatSDPA(
                dim=config.gizli_boyut,
                num_heads=config.dikkat_baslik_sayisi,
                dropout=config.dropout,
            )
        else:
            # Standart MultiheadAttention wrapper
            self.dikkat = nn.MultiheadAttention(
                embed_dim=config.gizli_boyut,
                num_heads=config.dikkat_baslik_sayisi,
                dropout=config.dropout,
                batch_first=True,
            )

        # 3. Normalizasyon 2
        if config.norm_turu == "rmsnorm":
            self.norm2 = RMSNorm(config.gizli_boyut)
        else:
            self.norm2 = nn.LayerNorm(config.gizli_boyut)

        # 4. FFN Katmanı
        if config.ffn_turu == "swiglu":
            # Parametrik denge için 8/3 d
            swiglu_dim = int(config.gizli_boyut * 8 / 3)
            self.ffn = SwiGLU(
                in_features=config.gizli_boyut,
                hidden_features=swiglu_dim,
                dropout=config.dropout,
            )
        else:
            self.ffn = GELUFFN(
                in_features=config.gizli_boyut,
                hidden_features=config.ileri_besleme_boyutu,
                dropout=config.dropout,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm Dikkat
        norm_x1 = self.norm1(x)
        if isinstance(self.dikkat, nn.MultiheadAttention):
            attn_out, _ = self.dikkat(norm_x1, norm_x1, norm_x1)
        else:
            attn_out = self.dikkat(norm_x1)

        x = x + attn_out

        # Pre-Norm FFN
        x = x + self.ffn(self.norm2(x))
        return x
