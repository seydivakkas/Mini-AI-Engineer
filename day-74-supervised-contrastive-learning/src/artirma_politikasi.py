"""
Supervised Contrastive Learning (SupCon) Çift Görünüm Artırma Boru Hattı
-----------------------------------------------------------------------
Aynı sınıfa ait çoklu örnekler ve stokastik artırma görünümleri üreten modül.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple
import torch
import torchvision.transforms as T
from PIL import Image


class SupConArtirici:
    """
    SupCon için 2 görünümlü dönüşüm boru hattı:
    - RandomResizedCrop
    - RandomHorizontalFlip
    - ColorJitter
    - RandomGrayscale
    - Normalizasyon
    """
    def __init__(self, goruntu_boyutu: int = 32, jitter_gucu: float = 0.5):
        self.goruntu_boyutu = goruntu_boyutu
        
        renk_jitter = T.ColorJitter(
            brightness=0.8 * jitter_gucu,
            contrast=0.8 * jitter_gucu,
            saturation=0.8 * jitter_gucu,
            hue=0.2 * jitter_gucu
        )
        
        self.donusum = T.Compose([
            T.RandomResizedCrop(goruntu_boyutu, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([renk_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])

    def __call__(self, x: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.donusum(x), self.donusum(x)


class TensorSupConArtirici:
    """
    Tensör batch'leri için hızlı GPU/CPU uyumlu artırıcı.
    """
    def __init__(self, goruntu_boyutu: int = 32):
        self.goruntu_boyutu = goruntu_boyutu
        self.donusum = T.Compose([
            T.RandomResizedCrop(goruntu_boyutu, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
        ])

    def cift_uret(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aynı batch'ten iki farklı pertürbe edilmiş görünüm üretir."""
        gurultu_1 = torch.randn_like(x) * 0.04
        gurultu_2 = torch.randn_like(x) * 0.04
        v1 = torch.clamp(self.donusum(x) + gurultu_1, 0.0, 1.0)
        v2 = torch.clamp(self.donusum(x) + gurultu_2, 0.0, 1.0)
        return v1, v2
