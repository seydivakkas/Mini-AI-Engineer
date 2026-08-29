"""
LoRA (Low-Rank Adaptation) Doğrusal Katmanı Modülü (Day 176 - FAZ 9).
Dondurulmuş orijinal ağırlığa (W_0) düşük dereceli B*A matris çarpımını ekler: W = W_0 + (alpha/r)*B*A
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA Doğrusal Katmanı (Hu et al., 2021).
    W_0 kilitlenir (requires_grad=False).
    A matrisi Gaussian N(0, 1/r), B matrisi 0 olarak başlatılır.
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # 1. Dondurulmuş Temel Katman (W_0)
        self.base_layer = linear_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # 2. Düşük Dereceli Matrisler A ve B
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        # A matrisini Kaiming/Gaussian başlat, B sıfır kalsın
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        y = x W_0^T + (x A^T B^T) * scaling
        """
        base_out = self.base_layer(x)
        # LoRA yolu
        lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + lora_out * self.scaling

    def agirliklari_birlestir(self) -> torch.Tensor:
        """W = W_0 + (alpha/r) * B * A (İnferansta sıfır gecikme için)"""
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        return self.base_layer.weight.data + delta_w
