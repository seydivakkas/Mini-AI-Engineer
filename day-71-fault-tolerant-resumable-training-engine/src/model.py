"""
Kompakt Vision Ağ Mimarisi
==========================
Çökmeye dayanıklı eğitim motoru deneyleri için optimize edilmiş,
Residual bağlantılara sahip kompakt Evrişimli Sinir Ağı (VisionNet).
"""

import torch
import torch.nn as nn


class KompaktVisionNet(nn.Module):
    """Deney modeli: 3 kanallı görsel girdilerini sınıflandıran Residual Vision Ağı."""

    def __init__(self, girdi_kanali: int = 3, sinif_sayisi: int = 5, taban_kanal: int = 32) -> None:
        super().__init__()
        self.giris = nn.Sequential(
            nn.Conv2d(girdi_kanali, taban_kanal, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(taban_kanal),
            nn.ReLU(inplace=True)
        )

        self.blok1 = nn.Sequential(
            nn.Conv2d(taban_kanal, taban_kanal * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(taban_kanal * 2),
            nn.ReLU(inplace=True)
        )

        self.blok2 = nn.Sequential(
            nn.Conv2d(taban_kanal * 2, taban_kanal * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(taban_kanal * 4),
            nn.ReLU(inplace=True)
        )

        self.havuz = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(taban_kanal * 4, sinif_sayisi)

        self._ilklendir()

    def _ilklendir(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.giris(x)
        x = self.blok1(x)
        x = self.blok2(x)
        x = self.havuz(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
