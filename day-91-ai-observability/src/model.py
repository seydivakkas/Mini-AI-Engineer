"""
Day 91: Canlı Çıkarım Yapan Derin Öğrenme Modeli ve Temsil Katmanı
-----------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionModelObservability(nn.Module):
    """
    Gözlemlenebilirlik deneyleri için optimize edilmiş, hem sınıf logit'lerini
    hem de gizli öznitelik vektörlerini (embeddings) döndürebilen CNN mimarisi.
    """

    def __init__(self, giris_kanali: int = 3, sinif_sayisi: int = 10, gizli_boyut: int = 64):
        super().__init__()
        self.giris_kanali = giris_kanali
        self.sinif_sayisi = sinif_sayisi
        self.gizli_boyut = gizli_boyut

        self.omurga = nn.Sequential(
            nn.Conv2d(giris_kanali, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            nn.Conv2d(64, gizli_boyut, kernel_size=3, padding=1),
            nn.BatchNorm2d(gizli_boyut),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.siniflandirici = nn.Sequential(
            nn.Linear(gizli_boyut, gizli_boyut // 2),
            nn.ReLU(inplace=True),
            nn.Linear(gizli_boyut // 2, sinif_sayisi),
        )

    def forward(self, x: torch.Tensor, ozellik_dondur: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        ozellikler = self.omurga(x)
        ozellik_vektoru = torch.flatten(ozellikler, 1)
        logitler = self.siniflandirici(ozellik_vektoru)

        if ozellik_dondur:
            return logitler, ozellik_vektoru
        return logitler

    def tahmin_et(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tahmin sınıfı, softmax olasılık dağılımı ve maksimum güven skorunu döner.
        """
        self.eval()
        with torch.no_grad():
            logitler, ozellikler = self.forward(x, ozellik_dondur=True)
            olasiliklar = F.softmax(logitler, dim=-1)
            guvenler, siniflar = torch.max(olasiliklar, dim=-1)
        return siniflar, olasiliklar, guvenler
