"""
SupCon Model Mimarisi (Kodlayıcı + Projeksiyon Kafası + Doğrusal Sınıflandırıcı)
--------------------------------------------------------------------------------
Supervised Contrastive Learning için 2 aşamalı (Stage 1 ve Stage 2) model yapısı.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemelKodlayici(nn.Module):
    """
    Görsel omurga ağı f(x) = h.
    Görüntü tensörlerini yüksek boyutlu temsil vektörüne (h) dönüştürür.
    """
    def __init__(self, giris_kanali: int = 3, temsil_boyutu: int = 128):
        super().__init__()
        self.temsil_boyutu = temsil_boyutu
        
        self.ozellik_cikarici = nn.Sequential(
            # Blok 1: 32x32 -> 16x16
            nn.Conv2d(giris_kanali, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Blok 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Blok 3: 8x8 -> 1x1
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.projeksiyon_fc = nn.Sequential(
            nn.Linear(128, temsil_boyutu),
            nn.BatchNorm1d(temsil_boyutu)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        c = self.ozellik_cikarici(x).view(b, -1)
        h = self.projeksiyon_fc(c)
        return h


class ProjeksiyonKafasi(nn.Module):
    """
    Non-lineer Projeksiyon Kafası g(h) = z.
    z = W2 * ReLU(W1 * h), L2 normalize edilmiş.
    """
    def __init__(self, temsil_boyutu: int = 128, projeksiyon_boyutu: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(temsil_boyutu, temsil_boyutu, bias=False),
            nn.BatchNorm1d(temsil_boyutu),
            nn.ReLU(inplace=True),
            nn.Linear(temsil_boyutu, projeksiyon_boyutu, bias=False),
            nn.BatchNorm1d(projeksiyon_boyutu, affine=False)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.mlp(h)
        z = F.normalize(z, p=2, dim=1)
        return z


class DogrusalSiniflandirici(nn.Module):
    """
    Stage 2 için Dondurulmuş (Frozen) h temsilleri üzerine kurulan doğrusal sınıflandırıcı.
    """
    def __init__(self, temsil_boyutu: int = 128, sinif_sayisi: int = 5):
        super().__init__()
        self.fc = nn.Linear(temsil_boyutu, sinif_sayisi)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


class SupConModeli(nn.Module):
    """
    Uçtan uca Supervised Contrastive Model: f(.) + g(.) + c(.)
    """
    def __init__(
        self,
        giris_kanali: int = 3,
        temsil_boyutu: int = 128,
        projeksiyon_boyutu: int = 64,
        sinif_sayisi: int = 5
    ):
        super().__init__()
        self.kodlayici = TemelKodlayici(giris_kanali, temsil_boyutu)
        self.projeksiyon = ProjeksiyonKafasi(temsil_boyutu, projeksiyon_boyutu)
        self.siniflandirici = DogrusalSiniflandirici(temsil_boyutu, sinif_sayisi)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 1: (h, z) çiftini döner."""
        h = self.kodlayici(x)
        z = self.projeksiyon(h)
        return h, z

    def siniflandir(self, x: torch.Tensor) -> torch.Tensor:
        """Stage 2: Girdiden doğrudan sınıf logits çıktısı döner."""
        with torch.no_grad():
            h = self.kodlayici(x)
        logits = self.siniflandirici(h)
        return logits

    def temsil_cikar(self, x: torch.Tensor) -> torch.Tensor:
        """Yalnızca temel h temsil vektörünü döner."""
        return self.kodlayici(x)
