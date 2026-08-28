"""
Yuksek Performansli Gorsel Veri Artirici (Albumentations Augmentation Engine)
===========================================================================
C++ ve OpenCV backend'i ile optimize edilmis, PyTorch torchvision'a kiyasla
3x-5x daha hizli calisan moduler goruntu donusturme ve artirma boru hatti.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class YuksekPerformansArtirici:
    """
    Albumentations tabanli yuksek verimli egitim ve dogrulama veri artirma motoru.
    """

    def __init__(
        self,
        hedef_boyut: Tuple[int, int] = (64, 64),
        ortalama: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        standart_sapma: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        artirma_olasilik: float = 0.5
    ) -> None:
        self.hedef_boyut = hedef_boyut
        self.ortalama = ortalama
        self.standart_sapma = standart_sapma
        self.artirma_olasilik = artirma_olasilik

        self.egitim_donusumu = self._egitim_donusumu_olustur()
        self.dogrulama_donusumu = self._dogrulama_donusumu_olustur()

    def _egitim_donusumu_olustur(self) -> A.Compose:
        """Kapsamlı ve C++ hızında çalışan eğitim veri artırma zinciri."""
        H, W = self.hedef_boyut
        return A.Compose([
            A.RandomResizedCrop(size=(H, W), scale=(0.8, 1.0), p=1.0),
            A.HorizontalFlip(p=self.artirma_olasilik),
            A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent=(-0.06, 0.06), p=self.artirma_olasilik),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=self.artirma_olasilik),
            A.GaussNoise(p=self.artirma_olasilik * 0.5),
            A.Normalize(mean=self.ortalama, std=self.standart_sapma),
            ToTensorV2()
        ])

    def _dogrulama_donusumu_olustur(self) -> A.Compose:
        """Deterministik doğrulama ve çıkarım dönüşüm zinciri."""
        H, W = self.hedef_boyut
        return A.Compose([
            A.Resize(height=H, width=W),
            A.Normalize(mean=self.ortalama, std=self.standart_sapma),
            ToTensorV2()
        ])

    def egitim_donustur(self, gorsel_np: np.ndarray) -> torch.Tensor:
        """
        NumPy HWC [0, 255] uint8 dizisini artırır ve CHW float32 tensörüne dönüştürür.
        """
        if not isinstance(gorsel_np, np.ndarray):
            raise TypeError("Girdi NumPy dizisi (ndarray) olmalidir.")
        sonuc = self.egitim_donusumu(image=gorsel_np)
        return sonuc["image"]

    def dogrulama_donustur(self, gorsel_np: np.ndarray) -> torch.Tensor:
        """Doğrulama dönüşümü uygular."""
        if not isinstance(gorsel_np, np.ndarray):
            raise TypeError("Girdi NumPy dizisi (ndarray) olmalidir.")
        sonuc = self.dogrulama_donusumu(image=gorsel_np)
        return sonuc["image"]
