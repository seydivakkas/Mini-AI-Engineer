"""
Triplet Metric Learning Öznitelik Çıkarıcı Ağ Mimarisi
------------------------------------------------------
Görsel girdileri L2-normalize edilmiş düşük boyutlu metrik uzayına projekte eden omurga.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetrikOznitelikAgi(nn.Module):
    """
    Metrik Öğrenimi için Evrişimli Temsil Ağı f(x) -> e.
    Çıktı vektörleri birim hiperküre üzerine L2 normalize edilir (||e||_2 = 1.0).
    """
    def __init__(self, giris_kanali: int = 3, gomulme_boyutu: int = 64):
        super().__init__()
        self.gomulme_boyutu = gomulme_boyutu
        
        self.omurga = nn.Sequential(
            # Blok 1: 32x32 -> 16x16
            nn.Conv2d(giris_kanali, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Blok 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Blok 3: 8x8 -> 1x1
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.projeksiyon = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, gomulme_boyutu, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        c = self.omurga(x).view(b, -1)
        e = self.projeksiyon(c)
        # Metrik uzayı için L2 normalizasyon
        e_norm = F.normalize(e, p=2, dim=1)
        return e_norm
