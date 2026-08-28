"""
Çıkarım Modeli Mimarisi (Vision Classifier)
-------------------------------------------
Dinamik batching çıkarım motorunda koşturulacak derin görme modeli.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionClassifier(nn.Module):
    """
    GPU tensör çekirdekleri için optimize edilmiş 4 katmanlı konvolüsyonel sınıflandırıcı.
    """
    def __init__(self, giris_kanali: int = 3, sinif_sayisi: int = 10, taban_kanal: int = 32):
        super().__init__()
        self.giris_kanali = giris_kanali
        self.sinif_sayisi = sinif_sayisi

        self.ozellik_cikarici = nn.Sequential(
            nn.Conv2d(giris_kanali, taban_kanal, kernel_size=3, padding=1),
            nn.BatchNorm2d(taban_kanal),
            nn.ReLU(),
            nn.Conv2d(taban_kanal, taban_kanal * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(taban_kanal * 2),
            nn.ReLU(),
            nn.Conv2d(taban_kanal * 2, taban_kanal * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(taban_kanal * 4),
            nn.ReLU(),
            nn.Conv2d(taban_kanal * 4, taban_kanal * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(taban_kanal * 8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.siniflandirici = nn.Linear(taban_kanal * 8, sinif_sayisi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.ozellik_cikarici(x).flatten(1)
        return self.siniflandirici(feat)
