"""
NeRF Derin MLP Modeli (Day 178 - FAZ 9).
3D konum x ve bakış açısı d alıp hacimsel yoğunluk sigma ve RGB renk c üretir.
"""

from typing import Tuple, List
import torch
import torch.nn as nn
from .pozisyonel_kodlayici import PozisyonelKodlayici


class NeRFModeli(nn.Module):
    """
    NeRF (Neural Radiance Field) 8 Katmanlı MLP Ağı (Mildenhall et al., 2020).
    Konum (xyz) -> Yoğunluk (sigma) ve Özellik Vektörü
    Özellik Vektörü + Yön (dir) -> Görüş Açısına Bağımlı RGB Rengi
    """

    def __init__(
        self,
        D: int = 8,
        hidden_dim: int = 128,
        pos_frequencies: int = 10,
        dir_frequencies: int = 4,
        skips: List[int] = [4],
    ):
        super().__init__()
        self.D = D
        self.skips = skips
        self.pos_encoder = PozisyonelKodlayici(in_dims=3, num_frequencies=pos_frequencies, include_input=True)
        self.dir_encoder = PozisyonelKodlayici(in_dims=3, num_frequencies=dir_frequencies, include_input=True)

        in_dim_pos = self.pos_encoder.out_dim
        in_dim_dir = self.dir_encoder.out_dim

        # 1. Konum İşleme Katmanları (Yoğunluk sigma için 8 katman)
        self.pts_linears = nn.ModuleList()
        for i in range(D):
            if i == 0:
                self.pts_linears.append(nn.Linear(in_dim_pos, hidden_dim))
            elif i in self.skips:
                self.pts_linears.append(nn.Linear(hidden_dim + in_dim_pos, hidden_dim))
            else:
                self.pts_linears.append(nn.Linear(hidden_dim, hidden_dim))

        self.sigma_linear = nn.Linear(hidden_dim, 1)
        self.feature_linear = nn.Linear(hidden_dim, hidden_dim)

        # 2. Yön İşleme Katmanları (RGB renk için)
        self.views_linears = nn.ModuleList([nn.Linear(hidden_dim + in_dim_dir, hidden_dim // 2)])
        self.rgb_linear = nn.Linear(hidden_dim // 2, 3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pts: torch.Tensor, views: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        pts:   [..., 3] (3D Uzay Noktaları)
        views: [..., 3] (3D Bakış Yönü Birim Vektörü)
        Döner: (rgb [..., 3], sigma [..., 1])
        """
        pts_encoded = self.pos_encoder(pts)
        views_encoded = self.dir_encoder(views)

        h = pts_encoded
        for i, l in enumerate(self.pts_linears):
            if i in self.skips:
                h = torch.cat([h, pts_encoded], dim=-1)
            h = self.relu(l(h))

        # Hacimsel Yoğunluk sigma (Pozitif olması için ReLU)
        sigma = self.relu(self.sigma_linear(h))
        feature = self.feature_linear(h)

        # Görüş Açısına Bağımlı RGB Rengi (0..1 aralığında Sigmoid)
        h_views = torch.cat([feature, views_encoded], dim=-1)
        for l in self.views_linears:
            h_views = self.relu(l(h_views))
        rgb = self.sigmoid(self.rgb_linear(h_views))

        return rgb, sigma
