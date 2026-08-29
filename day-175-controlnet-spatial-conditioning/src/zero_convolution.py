"""
Sıfır Konvolüsyon (Zero-Convolution) Modülü (Day 175 - FAZ 9).
Ağırlıkları ve bias değerleri sıfır (W=0, b=0) ile başlatılan 1x1 konvolüsyon katmanı.
"""

import torch
import torch.nn as nn


class ZeroConv2d(nn.Module):
    """
    ControlNet Zero-Convolution Katmanı (Zhang & Agrawala, 2023).
    Eğitimin başlangıcında y = 0 çıktısı vererek ana difüzyon modelinin
    orijinal yeteneklerini hiçbir zararlı gürültü vermeden korur.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        # Ağırlık ve bias'ları sıfırla
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_channels, H, W] -> y: [B, out_channels, H, W]"""
        return self.conv(x)
