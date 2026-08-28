"""
Day 93: Değerlendirmeye Tabi Tutulan Derin Öğrenme Vision Sınıflandırıcı Mimarisi
--------------------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FinalVisionClassifier(nn.Module):
    """
    Kapsamlı doğrulama, dilim analizi ve Model Card üretimi için
    özelleştirilmiş Vision Sınıflandırıcı derin öğrenme modeli.
    """

    def __init__(self, giris_kanali: int = 3, sinif_sayisi: int = 10, taban_filtre: int = 32):
        super().__init__()
        self.giris_kanali = giris_kanali
        self.sinif_sayisi = sinif_sayisi

        self.ozellik_cikarici = nn.Sequential(
            nn.Conv2d(giris_kanali, taban_filtre, kernel_size=3, padding=1),
            nn.BatchNorm2d(taban_filtre),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(taban_filtre, taban_filtre * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(taban_filtre * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(taban_filtre * 2, taban_filtre * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(taban_filtre * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.baslik = nn.Sequential(
            nn.Linear(taban_filtre * 4, taban_filtre * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(taban_filtre * 2, sinif_sayisi),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ozellikler = self.ozellik_cikarici(x)
        duz = torch.flatten(ozellikler, 1)
        return self.baslik(duz)

    def tahmin_et(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Softmax olasılıkları ve en olası sınıf indekslerini döner."""
        self.eval()
        with torch.no_grad():
            logitler = self.forward(x)
            olasiliklar = F.softmax(logitler, dim=-1)
            tahmin_siniflar = torch.argmax(olasiliklar, dim=-1)
        return tahmin_siniflar, olasiliklar
