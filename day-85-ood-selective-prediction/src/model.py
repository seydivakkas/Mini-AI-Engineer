"""
Vision OOD Modeli ve Öznitelik Çıkarıcı
--------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class VisionOODModeli(nn.Module):
    """
    3 Aşamalı Konvolüsyonel Vision Modeli
    """
    def __init__(self, giris_kanali: int = 3, sinif_sayisi: int = 10, taban_kanal: int = 32):
        super().__init__()
        self.giris_kanali = giris_kanali
        self.sinif_sayisi = sinif_sayisi

        self.omurga = nn.Sequential(
            nn.Conv2d(giris_kanali, taban_kanal, kernel_size=3, padding=1),
            nn.BatchNorm2d(taban_kanal),
            nn.ReLU(),
            nn.Conv2d(taban_kanal, taban_kanal * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(taban_kanal * 2),
            nn.ReLU(),
            nn.Conv2d(taban_kanal * 2, taban_kanal * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(taban_kanal * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.siniflandirici = nn.Sequential(
            nn.Linear(taban_kanal * 4, taban_kanal * 2),
            nn.ReLU(),
            nn.Linear(taban_kanal * 2, sinif_sayisi)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.omurga(x).flatten(1)
        return self.siniflandirici(x)

    @classmethod
    def logit_cikar(cls, model: nn.Module, veri_loader: DataLoader, cihaz: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        model = model.to(cihaz).eval()
        tum_logitler = []
        tum_etiketler = []

        with torch.no_grad():
            for x, y in veri_loader:
                x = x.to(cihaz)
                logitler = model(x)
                tum_logitler.append(logitler.cpu())
                tum_etiketler.append(y)

        return torch.cat(tum_logitler, dim=0), torch.cat(tum_etiketler, dim=0)
