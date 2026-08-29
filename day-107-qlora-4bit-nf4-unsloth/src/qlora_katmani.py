"""
QLoRA Katmanı ve Unsloth Tarzı Füzyonlu Hızlı Autograd Modülü (Day 107).
4-bit dondurulmuş (frozen) NF4 ana ağırlıklar ve eğitilebilir LoRA adaptörleri.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.autograd import Function

from .nf4_kuantizasyon import NF4Kuantizator, DoubleQuantization


class HizliQLoRAAutograd(Function):
    """
    Unsloth tarzı optimize edilmiş, ara aktivasyon bellek tüketimini minimize eden
    özel QLoRA ileri ve geri geçiş (Autograd) fonksiyonu.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_dequant: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, w_dequant, lora_A, lora_B, bias)
        ctx.scaling = scaling

        # 1. Ana Model Çıktısı (Dequantized FP16/FP32 Base Weight)
        out_base = torch.matmul(x, w_dequant.t())

        # 2. LoRA Adaptör Çıktısı: scaling * (X @ A^T) @ B^T
        x_A = torch.matmul(x, lora_A.t())  # [..., r]
        out_lora = torch.matmul(x_A, lora_B.t()) * scaling

        out = out_base + out_lora
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, w_dequant, lora_A, lora_B, bias = ctx.saved_tensors
        scaling = ctx.scaling

        # Şekil düzenlemeleri (Düzleştirme: [N, in_features] veya [B*S, in_features])
        x_reshaped = x.reshape(-1, x.shape[-1])
        grad_reshaped = grad_output.reshape(-1, grad_output.shape[-1])

        # 1. dL/dB = scaling * grad_output^T @ (X @ A^T)
        x_A = torch.matmul(x_reshaped, lora_A.t())  # [N, r]
        grad_lora_B = scaling * torch.matmul(grad_reshaped.t(), x_A)

        # 2. dL/dA = scaling * (grad_output @ B)^T @ X
        grad_B = torch.matmul(grad_reshaped, lora_B)  # [N, r]
        grad_lora_A = scaling * torch.matmul(grad_B.t(), x_reshaped)

        # 3. dL/dX = grad_output @ W + scaling * (grad_output @ B) @ A
        grad_x_base = torch.matmul(grad_reshaped, w_dequant)
        grad_x_lora = scaling * torch.matmul(grad_B, lora_A)
        grad_x = (grad_x_base + grad_x_lora).reshape(x.shape)

        # 4. dL/dBias
        grad_bias = grad_reshaped.sum(dim=0) if bias is not None else None

        # w_dequant dondurulmuştur (requires_grad=False)
        return grad_x, None, grad_lora_A, grad_lora_B, None, grad_bias


class QLoRALinear(nn.Module):
    """
    4-bit NF4 Kuantize Edilmiş ve Double Quantization ile Sıkıştırılmış QLoRA Katmanı.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        block_size: int = 64,
        double_quant: bool = True,
        bias: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.block_size = block_size
        self.double_quant = double_quant
        self.device = device

        self.kuantizator = NF4Kuantizator(block_size=block_size, device=device)
        self.dq_motoru = DoubleQuantization(block_size_2=256)

        # 1. Dondurulmuş 4-bit Ağırlık Saklayıcıları (Non-trainable Buffers)
        self.register_buffer(
            "q_weight",
            torch.zeros((out_features, in_features), dtype=torch.uint8, device=device),
        )
        self.register_buffer("c1", torch.zeros(1, dtype=torch.float32, device=device))
        self.register_buffer("c1_int8", torch.zeros(1, dtype=torch.uint8, device=device))
        self.register_buffer("c2", torch.zeros(1, dtype=torch.float32, device=device))
        self.register_buffer("c1_min", torch.zeros(1, dtype=torch.float32, device=device))

        # 2. Eğitilebilir LoRA Adaptörleri
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty((r, in_features), device=device))
            self.lora_B = nn.Parameter(torch.empty((out_features, r), device=device))
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
            self.sifirla_parametreler()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter("bias", None)

    def sifirla_parametreler(self):
        """LoRA adaptör parametrelerini ilklendirir (A: Kaiming Uniform, B: Sıfır)."""
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def agirliklari_yukle_ve_kuantize_et(self, fp_weight: torch.Tensor):
        """FP16/FP32 ana ağırlıkları 4-bit NF4 ve Double Quantization ile kuantize eder."""
        assert fp_weight.shape == (self.out_features, self.in_features)
        q_w, c1_scale, _ = self.kuantizator.kuantize_et(fp_weight)
        self.q_weight.copy_(q_w)

        if self.double_quant:
            c1_int8, c2, c1_min = self.dq_motoru.c1_sikistir(c1_scale)
            self.c1_int8 = c1_int8.to(self.device)
            self.c2 = c2.to(self.device)
            self.c1_min = c1_min.to(self.device)
        else:
            self.c1 = c1_scale.to(self.device)

    def agirligi_dekuantize_et(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Çıkarım ve geri yayılım için 4-bit NF4 ağırlığı çözer."""
        if self.double_quant:
            c1_cozulmus = self.dq_motoru.c1_coz(self.c1_int8, self.c2, self.c1_min)
            return self.kuantizator.dekuantize_et(
                self.q_weight, c1_cozulmus, (self.out_features, self.in_features), dtype=dtype
            )
        else:
            return self.kuantizator.dekuantize_et(
                self.q_weight, self.c1, (self.out_features, self.in_features), dtype=dtype
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_deq = self.agirligi_dekuantize_et(dtype=x.dtype)

        if self.r > 0:
            x_drop = self.lora_dropout(x)
            return HizliQLoRAAutograd.apply(
                x_drop, w_deq, self.lora_A, self.lora_B, self.scaling, self.bias
            )
        else:
            out = torch.matmul(x, w_deq.t())
            if self.bias is not None:
                out = out + self.bias
            return out
