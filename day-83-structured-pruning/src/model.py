"""
Budanabilir Çok Aşamalı Vision CNN Mimarisi
-------------------------------------------
Yapısal kanal/filtre budama (Structured Filter Pruning) için optimize edilmiş,
dinamik kanal konfigürasyonunu ve katman dikişlerini (Layer Stitching) destekleyen model.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import List, Tuple
import torch
import torch.nn as nn


class BudanabilirVisionCNN(nn.Module):
    """
    Kanal listesi [c1, c2, c3] dinamik olarak güncellenebilen evrişimli model.
    """
    def __init__(
        self,
        giris_kanali: int = 3,
        sinif_sayisi: int = 10,
        kanallar: List[int] = [32, 64, 128]
    ):
        super().__init__()
        self.giris_kanali = giris_kanali
        self.sinif_sayisi = sinif_sayisi
        self.kanallar = list(kanallar)

        c1, c2, c3 = self.kanallar

        # Aşama 1
        self.conv1 = nn.Conv2d(giris_kanali, c1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.relu1 = nn.ReLU(inplace=True)

        # Aşama 2
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Aşama 3
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c3)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        # Sınıflandırıcı Kafa
        self.fc = nn.Linear(c3, sinif_sayisi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        return self.fc(x)
