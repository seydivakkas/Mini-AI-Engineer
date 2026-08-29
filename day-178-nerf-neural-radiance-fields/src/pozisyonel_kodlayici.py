"""
Pozisyonel Kodlayıcı Modülü (Day 178 - FAZ 9).
Mildenhall et al. (2020) NeRF Fourier Frekans Dönüşümü:
gamma(p) = [sin(2^0 pi p), cos(2^0 pi p), ..., sin(2^{L-1} pi p), cos(2^{L-1} pi p)]
"""

import math
import torch
import torch.nn as nn


class PozisyonelKodlayici(nn.Module):
    """
    Sürekli 3D uzay koordinatlarını (x, y, z) ve 3D bakış yönlerini (dx, dy, dz)
    yüksek frekanslı Fourier uzayına yansıtarak yüksek frekanslı doku detaylarını yakalar.
    """

    def __init__(self, in_dims: int = 3, num_frequencies: int = 10, include_input: bool = True):
        super().__init__()
        self.in_dims = in_dims
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # 2^0, 2^1, ..., 2^{L-1} frekans bantları
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_frequencies - 1, num_frequencies)

        # Çıktı boyutu: in_dims * (2 * L + (1 if include_input else 0))
        self.out_dim = in_dims * (2 * num_frequencies + (1 if include_input else 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., in_dims]
        Döner: [..., out_dim]
        """
        bands = self.freq_bands.to(x.device)
        out = [x] if self.include_input else []

        for freq in bands:
            out.append(torch.sin(x * (freq * math.pi)))
            out.append(torch.cos(x * (freq * math.pi)))

        return torch.cat(out, dim=-1)
