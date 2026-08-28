"""
Modern Regüler Vision Ağ Mimarisi
=================================
Mixup, CutMix ve Label Smoothing deneyleri için optimize edilmiş,
Residual bağlantılara sahip kompakt Evrişimli Sinir Ağı.
"""

import torch
import torch.nn as nn


class ResidualBlok(nn.Module):
    def __init__(self, kanallar: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(kanallar, kanallar, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(kanallar)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(kanallar, kanallar, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(kanallar)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        artik = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += artik
        return self.relu(out)


class ModernRegulerVisionNet(nn.Module):
    """Deney modeli: 3 kanallı görsel girdilerini sınıflandıran Residual Vision Ağı."""

    def __init__(self, girdi_kanali: int = 3, sinif_sayisi: int = 5, taban_kanal: int = 32) -> None:
        super().__init__()
        self.kok = nn.Sequential(
            nn.Conv2d(girdi_kanali, taban_kanal, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(taban_kanal),
            nn.ReLU(inplace=True)
        )

        self.katman1 = nn.Sequential(
            nn.Conv2d(taban_kanal, taban_kanal * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(taban_kanal * 2),
            nn.ReLU(inplace=True),
            ResidualBlok(taban_kanal * 2)
        )

        self.katman2 = nn.Sequential(
            nn.Conv2d(taban_kanal * 2, taban_kanal * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(taban_kanal * 4),
            nn.ReLU(inplace=True),
            ResidualBlok(taban_kanal * 4)
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
        x = self.kok(x)
        x = self.katman1(x)
        x = self.katman2(x)
        x = self.havuz(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
