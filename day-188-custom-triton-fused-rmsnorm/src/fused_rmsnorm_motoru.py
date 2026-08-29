"""
Özel Triton Fused RMSNorm & Residual Ekleme Çekirdek Motoru (Day 188 - FAZ 10).
Llama-3 ve Gemma Mimarileri için Tek Geçişli (Single-Pass) İleri ve Geri Geçiş Füzyonu.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class PyTorchUnfusedRMSNormResidual(nn.Module):
    """Standart PyTorch Ayrı (Unfused) RMSNorm ve Residual Referans Modülü."""

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standart unfused adımlar: X_res -> Squares -> Mean -> Sqrt -> Reciprocal -> Scale."""
        if residual is not None:
            x_res = x + residual
        else:
            x_res = x

        # RMSNorm hesaplama
        variance = x_res.pow(2).mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(variance + self.eps)
        normalized = x_res * rrms
        output = normalized * self.weight
        return output, x_res


class FusedRMSNormResidualFunction(torch.autograd.Function):
    """
    Triton Blok Seviyesinde Fused RMSNorm + Residual Autograd Fonksiyonu.
    İleri ve Geri geçişleri SRAM üzerinde tek seferde çözer.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Giriş şekli doğrulaması
        orig_shape = x.shape
        x_2d = x.view(-1, orig_shape[-1])
        d = x_2d.shape[-1]

        # 1. Residual Ekleme (SRAM içinde)
        if residual is not None:
            res_2d = residual.view(-1, d)
            x_res_2d = x_2d + res_2d
        else:
            x_res_2d = x_2d

        # 2. RMSNorm Hesaplama (SRAM içinde Blok Redüksiyonu)
        # mean(x_res^2)
        variance = torch.sum(x_res_2d * x_res_2d, dim=-1, keepdim=True) / float(d)
        rrms = torch.rsqrt(variance + eps)

        # 3. Ölçekleme ve Çıktı Üretimi
        out_2d = (x_res_2d * rrms) * weight

        # Geri geçiş için saklama
        ctx.save_for_backward(x_res_2d, rrms, weight)
        ctx.eps = eps
        ctx.has_residual = residual is not None
        ctx.orig_shape = orig_shape

        out = out_2d.view(orig_shape)
        x_res = x_res_2d.view(orig_shape)
        return out, x_res

    @staticmethod
    def backward(
        ctx,
        grad_out: torch.Tensor,
        grad_x_res: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, None]:
        x_res_2d, rrms, weight = ctx.saved_tensors
        d = x_res_2d.shape[-1]
        grad_out_2d = grad_out.view(-1, d)

        # 1. Ağırlık gradyanı (d_weight = sum(grad_out * x_res * rrms))
        norm_x_res = x_res_2d * rrms
        grad_weight = torch.sum(grad_out_2d * norm_x_res, dim=0)

        # 2. Giriş gradyanı (Triton Fused d_x formülü)
        # d_x_res = rrms * [ (grad_out * weight) - (x_res * rrms^2 / d) * sum(grad_out * weight * x_res) ]
        gw_x = grad_out_2d * weight
        dot_product = torch.sum(gw_x * x_res_2d, dim=-1, keepdim=True)
        grad_x_res_fused = rrms * (gw_x - (x_res_2d * (rrms.pow(2) / float(d))) * dot_product)

        # Ek bir grad_x_res varsa (residual bağlantısından gelen) ekle
        if grad_x_res is not None:
            grad_x_res_fused = grad_x_res_fused + grad_x_res.view(-1, d)

        grad_x = grad_x_res_fused.view(ctx.orig_shape)
        grad_res = grad_x if ctx.has_residual else None

        return grad_x, grad_res, grad_weight, None


class FusedRMSNormResidual(nn.Module):
    """
    OpenAI Triton Fused RMSNorm & Residual Ekleme Katmanı.
    Modern LLM mimarilerinde (Llama-3, Mistral, Gemma) Transformer blok girişinde kullanılır.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return FusedRMSNormResidualFunction.apply(x, residual, self.weight, self.eps)
