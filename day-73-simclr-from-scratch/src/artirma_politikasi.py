"""
SimCLR Çift Görünümlü Veri Artırma Politikası
--------------------------------------------
Aynı girdi görüntüsünden iki farklı stokastik artırma (view) üreten,
kontrastif öğrenim için optimize edilmiş dönüşüm boru hattı.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple
import torch
import torchvision.transforms as T
from PIL import Image


class SimCLRArtirmaDemetleyici:
    """
    SimCLR makalesinde (Chen et al., 2020) önerilen optimal artırma bileşimini uygular:
    1. Random Resized Crop (Ölçekleme ve Kırpma)
    2. Random Horizontal Flip (Yatay Çevirme)
    3. Color Jitter (Parlaklık, Kontrast, Doygunluk, Ton Değişimi)
    4. Random Grayscale (Gri Ton Dönüşümü)
    5. Gaussian Blur (Gauss Bulanıklaştırma)
    """
    def __init__(
        self,
        goruntu_boyutu: int = 32,
        jitter_gucu: float = 0.5,
        bulanıklastirma: bool = True
    ):
        self.goruntu_boyutu = goruntu_boyutu
        
        renk_jitter = T.ColorJitter(
            brightness=0.8 * jitter_gucu,
            contrast=0.8 * jitter_gucu,
            saturation=0.8 * jitter_gucu,
            hue=0.2 * jitter_gucu
        )
        
        donusumler = [
            T.RandomResizedCrop(goruntu_boyutu, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([renk_jitter], p=0.8),
            T.RandomGrayscale(p=0.2)
        ]
        
        if bulanıklastirma:
            donusumler.append(T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5))
            
        donusumler.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
        
        self.artirma_donusumu = T.Compose(donusumler)

    def __call__(self, x: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tek bir PIL görüntüsünden iki farklı artırılmış pozitif çift (x_i, x_j) döner.
        """
        gorunum_1 = self.artirma_donusumu(x)
        gorunum_2 = self.artirma_donusumu(x)
        return gorunum_1, gorunum_2


class TensorSimCLRArtirici:
    """
    Tensor formatındaki batch'lere GPU/CPU üzerinde doğrudan artırma uygular.
    """
    def __init__(self, goruntu_boyutu: int = 32):
        self.goruntu_boyutu = goruntu_boyutu
        self.donusum = T.Compose([
            T.RandomResizedCrop(goruntu_boyutu, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
        ])

    def cift_uret(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tensor batch'inden iki farklı pozitif görünüm üretir."""
        # Hafif renk/gürültü pertürbasyonu ekle
        gurultu_1 = torch.randn_like(x) * 0.05
        gurultu_2 = torch.randn_like(x) * 0.05
        
        v1 = torch.clamp(self.donusum(x) + gurultu_1, 0.0, 1.0)
        v2 = torch.clamp(self.donusum(x) + gurultu_2, 0.0, 1.0)
        return v1, v2
