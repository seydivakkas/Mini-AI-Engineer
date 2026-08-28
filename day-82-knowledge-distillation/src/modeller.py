"""
Öğretmen (Teacher) ve Öğrenci (Student) Model Mimarileri
-------------------------------------------------------
Öğretmen: Yüksek kapasiteli derin model (Büyük Alıcı Alan / Derinlik).
Öğrenci: Kenar cihazlar ve düşük gecikmeli çıkarım için optimize edilmiş kompakt model.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DerinKonvolusyonelOgretmen(nn.Module):
    """
    Yüksek Kapasiteli Öğretmen Modeli (Derin Evrişimli Bloklar + Residual)
    """
    def __init__(self, giris_kanali: int = 3, sinif_sayisi: int = 10, taban_kanal: int = 32):
        super().__init__()
        self.sinif_sayisi = sinif_sayisi

        self.giris_katmani = nn.Sequential(
            nn.Conv2d(giris_kanali, taban_kanal, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(taban_kanal),
            nn.GELU()
        )

        # 3 Aşamalı Derinlik
        self.asama1 = self._blok_yap(taban_kanal, taban_kanal * 2, katman_sayisi=3)      # 16x16
        self.asama2 = self._blok_yap(taban_kanal * 2, taban_kanal * 4, katman_sayisi=3)  # 8x8
        self.asama3 = self._blok_yap(taban_kanal * 4, taban_kanal * 8, katman_sayisi=3)  # 4x4

        self.havuzlama = nn.AdaptiveAvgPool2d((1, 1))
        self.siniflandirici = nn.Sequential(
            nn.Linear(taban_kanal * 8, taban_kanal * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(taban_kanal * 4, sinif_sayisi)
        )

    def _blok_yap(self, in_c: int, out_c: int, katman_sayisi: int) -> nn.Sequential:
        katmanlar = [
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        ]
        for _ in range(katman_sayisi - 1):
            katmanlar.extend([
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.GELU()
            ])
        return nn.Sequential(*katmanlar)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.giris_katmani(x)
        x = self.asama1(x)
        x = self.asama2(x)
        x = self.asama3(x)
        x = self.havuzlama(x).flatten(1)
        return self.siniflandirici(x)


class KompaktOgrenciModeli(nn.Module):
    """
    Hafif ve Kompakt Öğrenci Modeli (Düşük Parametre Sayısı & Hızlı Çıkarım)
    """
    def __init__(self, giris_kanali: int = 3, sinif_sayisi: int = 10, taban_kanal: int = 16):
        super().__init__()
        self.sinif_sayisi = sinif_sayisi

        self.omurga = nn.Sequential(
            nn.Conv2d(giris_kanali, taban_kanal, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(taban_kanal),
            nn.ReLU(),
            nn.Conv2d(taban_kanal, taban_kanal * 2, kernel_size=3, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(taban_kanal * 2),
            nn.ReLU(),
            nn.Conv2d(taban_kanal * 2, taban_kanal * 4, kernel_size=3, stride=2, padding=1), # 4x4
            nn.BatchNorm2d(taban_kanal * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.kafa = nn.Linear(taban_kanal * 4, sinif_sayisi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.omurga(x).flatten(1)
        return self.kafa(x)
