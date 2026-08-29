"""
BitNet b1.58: Sıfırdan 1.58-Bit ({-1, 0, 1}) Ternary LLM ve Matmul-Free Çıkarım Motoru (Day 261).
FAZ 14: Donanım Düzeyi Kernel Geliştirme, ASIC/NPU & 1-Bit LLM Açılış Modülü.
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_quantization_b158(w: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ağırlıkları {-1, 0, 1} kümesine (1.58-bit ternary) kuantize eder (BitNet b1.58).
    Ölçek: gamma = mean(|W|).
    W_tilde = Clip(Round(W / (gamma + eps)), -1, 1).
    Straight-Through Estimator (STE) ile gradyan akışı korunur.
    """
    gamma = torch.mean(torch.abs(w)) + eps
    w_scaled = w / gamma
    w_ternary = torch.clamp(torch.round(w_scaled), -1.0, 1.0)
    # STE: İleri yönde kuantize, geri yönde orijinal gradyan
    w_ste = w + (w_ternary - w).detach()
    return w_ste, gamma


def activation_quantization_int8(x: torch.Tensor, eps: float = 1e-5, b: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aktivasyonları 8-bit INT8 [-127, 127] aralığına kuantize eder.
    Ölçek: gamma_x = max(|X|).
    X_tilde = Clip(Round(X * Q_b / (gamma_x + eps)), -Q_b, Q_b).
    """
    q_b = 2 ** (b - 1) - 1  # 127
    gamma_x = torch.max(torch.abs(x), dim=-1, keepdim=True)[0] + eps
    x_scaled = x * (q_b / gamma_x)
    x_int8 = torch.clamp(torch.round(x_scaled), -q_b, q_b)
    # STE
    x_ste = x + (x_int8 - x).detach()
    return x_ste, gamma_x


class RMSNorm(nn.Module):
    """Kök Ortalama Kare Normalizasyonu (Root Mean Square Layer Normalization)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm


class BitLinear(nn.Module):
    """
    BitNet b1.58 1.58-Bit Ternary Doğrusal Katmanı.
    Pahalı kayan nokta (FP16/FP32) matris çarpımlarını INT8 toplama işlemlerine indirger.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)))
        self.norm = RMSNorm(in_features)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. RMSNorm
        x_norm = self.norm(x)

        # 2. 8-Bit Aktivasyon Kuantizasyonu
        x_quant, gamma_x = activation_quantization_int8(x_norm)

        # 3. 1.58-Bit Ağırlık Ternarizasyonu {-1, 0, 1}
        w_quant, gamma_w = weight_quantization_b158(self.weight)

        # 4. Matmul-Free Çarpım (Donanımda sadece toplama ağacı - Adder Tree)
        # Y = (X_int8 * W_ternary^T) * (gamma_x * gamma_w / 127)
        y = F.linear(x_quant, w_quant)
        scale = (gamma_x * gamma_w) / 127.0
        y = y * scale

        if self.bias is not None:
            y = y + self.bias
        return y


class BitNetAttention(nn.Module):
    """BitLinear Projeksiyonlu Çok Başlıklı Öz-Dikkat (Multi-Head Self-Attention)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = BitLinear(d_model, d_model)
        self.k_proj = BitLinear(d_model, d_model)
        self.v_proj = BitLinear(d_model, d_model)
        self.out_proj = BitLinear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)


class BitNetFFN(nn.Module):
    """BitLinear tabanlı İleri Beslemeli Ağ (SwiGLU Feed-Forward Network)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = BitLinear(d_model, d_ff)
        self.up_proj = BitLinear(d_model, d_ff)
        self.down_proj = BitLinear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class BitNetBlock(nn.Module):
    """1.58-Bit Ternary Transformer Bloğu."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = BitNetAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = BitNetFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class BitNetTransformer(nn.Module):
    """Sıfırdan 1.58-Bit BitNet b1.58 Büyük Dil Modeli (Ternary LLM)."""

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([BitNetBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = BitLinear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits
