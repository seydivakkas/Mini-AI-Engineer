"""
MiniViT-MoE v2 Katmanları ve Yönlendirici (Router) Modülü (Day 101).
Top-K Softmax Router, SwiGLU Uzmanları ve Yük Dengeleme (Load Balancing Loss) bileşenleri.
"""

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .konfigurasyon import MiniViTMoEConfig


class RMSNorm(nn.Module):
    """Root Mean Square Normalization (RMSNorm)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class ModernDikkatSDPA(nn.Module):
    """PyTorch 2.0 SDPA (FlashAttention-2) Çoklu Başlıklı Dikkat."""
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "Boyut başlık sayısına tam bölünmelidir."
        self.head_dim = dim // num_heads

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        drop_rate = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_rate, is_causal=False)
        return self.out_proj(out.transpose(1, 2).reshape(B, N, D))


class SwiGLUUzmani(nn.Module):
    """Bireysel SwiGLU Uzman Bloğu (Expert Feed-Forward Network)."""
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
        self.w_up = nn.Linear(in_features, hidden_features, bias=False)
        self.w_down = nn.Linear(hidden_features, in_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


class TopKRouter(nn.Module):
    """
    Top-K Softmax Yönlendirici (Router / Gating Network).
    Her token için en yüksek ağırlığa sahip k adet uzmanı seçer ve yük dengeleme kaybı üretir.
    """
    def __init__(
        self,
        dim: int,
        uzman_sayisi: int = 4,
        aktif_uzman_sayisi: int = 2,
        jitter_noise: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.uzman_sayisi = uzman_sayisi
        self.aktif_uzman_sayisi = min(aktif_uzman_sayisi, uzman_sayisi)
        self.jitter_noise = jitter_noise

        self.kapi_proj = nn.Linear(dim, uzman_sayisi, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Girdi: x -> [B, N, D]
        Çıktı:
          - secilen_uzmanlar: [B*N, k]
          - uzman_agirliklari: [B*N, k]
          - aux_loss: Skaler Yük Dengeleme Kaybı
        """
        B, N, D = x.shape
        flat_x = x.reshape(-1, D)  # [BN, D]

        logits = self.kapi_proj(flat_x)  # [BN, E]

        # Eğitim esnasında dengeli keşif için jitter gürültüsü
        if self.training and self.jitter_noise > 0.0:
            noise = torch.randn_like(logits) * self.jitter_noise
            logits = logits + noise

        router_probs = F.softmax(logits, dim=-1)  # [BN, E]

        # Top-K Uzman Seçimi
        topk_weights, topk_indices = torch.topk(router_probs, self.aktif_uzman_sayisi, dim=-1)

        # Ağırlıkların normalize edilmesi
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # Auxiliary Load Balancing Loss (Switch Transformer / Mixtral standardı)
        # f_i: Uzmana gönderilen token oranı
        # P_i: Uzmana atanan ortalama yönlendirici olasılığı
        # Loss = E * sum(f_i * P_i)
        tokens_per_expert = torch.zeros(self.uzman_sayisi, device=x.device)
        for i in range(self.aktif_uzman_sayisi):
            tokens_per_expert.scatter_add_(0, topk_indices[:, i], torch.ones_like(topk_indices[:, i], dtype=torch.float))
        f_i = tokens_per_expert / (flat_x.shape[0] * self.aktif_uzman_sayisi)
        P_i = router_probs.mean(dim=0)
        aux_loss = self.uzman_sayisi * torch.sum(f_i * P_i)

        return topk_indices, topk_weights, aux_loss


class MoEKatmani(nn.Module):
    """
    Sparse Mixture of Experts (MoE) Katmanı.
    Gelen token'ları yönlendiricinin seçtiği uzmanlara dağıtır ve ağırlıklı toplamını alır.
    """
    def __init__(self, config: MiniViTMoEConfig):
        super().__init__()
        self.config = config
        self.uzman_sayisi = config.uzman_sayisi
        self.aktif_uzman_sayisi = config.aktif_uzman_sayisi

        self.router = TopKRouter(
            dim=config.gizli_boyut,
            uzman_sayisi=config.uzman_sayisi,
            aktif_uzman_sayisi=config.aktif_uzman_sayisi,
            jitter_noise=config.router_jitter_noise,
        )

        # Her uzman için SwiGLU FFN bloğu
        swiglu_dim = int(config.gizli_boyut * 8 / 3)
        self.uzmanlar = nn.ModuleList([
            SwiGLUUzmani(
                in_features=config.gizli_boyut,
                hidden_features=swiglu_dim,
                dropout=config.dropout,
            )
            for _ in range(config.uzman_sayisi)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, N, D]
        B, N, D = x.shape
        flat_x = x.reshape(-1, D)  # [BN, D]

        topk_indices, topk_weights, aux_loss = self.router(x)  # [BN, k], [BN, k]

        cikti = torch.zeros_like(flat_x)  # [BN, D]

        # Her aktif uzman kademesi için hesaplama
        for k_idx in range(self.aktif_uzman_sayisi):
            expert_indices = topk_indices[:, k_idx]  # [BN]
            weights = topk_weights[:, k_idx].unsqueeze(-1)  # [BN, 1]

            for e_idx, uzman in enumerate(self.uzmanlar):
                mask = (expert_indices == e_idx)
                if mask.any():
                    token_grubu = flat_x[mask]
                    uzman_cikti = uzman(token_grubu)
                    cikti[mask] += weights[mask] * uzman_cikti

        return cikti.reshape(B, N, D), aux_loss


class MoETransformerBlok(nn.Module):
    """
    Pre-RMSNorm, FlashAttention/SDPA ve MoE FFN Katmanlı Modern Transformer Bloğu.
    """
    def __init__(self, config: MiniViTMoEConfig):
        super().__init__()
        self.config = config

        self.norm1 = RMSNorm(config.gizli_boyut) if config.norm_turu == "rmsnorm" else nn.LayerNorm(config.gizli_boyut)
        self.dikkat = ModernDikkatSDPA(
            dim=config.gizli_boyut,
            num_heads=config.dikkat_baslik_sayisi,
            dropout=config.dropout,
        )

        self.norm2 = RMSNorm(config.gizli_boyut) if config.norm_turu == "rmsnorm" else nn.LayerNorm(config.gizli_boyut)
        self.moe_katmani = MoEKatmani(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-Norm Attention
        norm_x1 = self.norm1(x)
        attn_out = self.dikkat(norm_x1)
        x = x + attn_out

        # Pre-Norm MoE FFN
        norm_x2 = self.norm2(x)
        moe_out, aux_loss = self.moe_katmani(norm_x2)
        x = x + moe_out

        return x, aux_loss
