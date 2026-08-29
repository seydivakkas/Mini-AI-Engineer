"""
adaLN-Zero (Adaptive Layer Normalization Zero) Modülü (Day 177 - FAZ 9).
Peebles & Xie (2023) DiT mimarisi için koşullandırma ve sıfır-başlatmalı modülasyon.
"""

from typing import Tuple
import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x tensörünü shift ve scale parametreleriyle modüle eder: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNZero(nn.Module):
    """
    adaLN-Zero Modülasyon Katmanı.
    Koşul vektöründen (c) 6 adet ölçek/kaydırma parametresi üretir:
    (gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2).
    alpha_1 ve alpha_2 parametreleri başlangıçta sıfır başlatılır (Zero-Init Identity Block).
    """

    def __init__(self, hidden_size: int, cond_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_size, 6 * hidden_size, bias=True)
        )
        # Sıfır başlatma: Başlangıçta Transformer bloğu birim fonksiyon (Identity) gibi davranır
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """6 adet modülasyon tensörü döner: gamma1, beta1, alpha1, gamma2, beta2, alpha2"""
        params = self.adaLN_modulation(c).chunk(6, dim=1)
        return params[0], params[1], params[2], params[3], params[4], params[5]
