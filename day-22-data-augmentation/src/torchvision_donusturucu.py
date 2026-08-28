"""torchvision.transforms ile PyTorch Tensör Seviyesinde Veri Çoğaltma.

Bu modül; PyTorch'un yerel torchvision.transforms kütüphanesini kullanarak
tensör seviyesinde doğrudan GPU üzerinde koşturulabilecek geometrik ve fotometrik
dönüşüm ardışık düzenlerini tanımlar.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torchvision.transforms as T


class TorchvisionDonusturucu:
    """torchvision dönüşüm ardışık düzenlerini yöneten sınıf."""

    def __init__(self, hedef_boyut: Tuple[int, int] = (64, 64)) -> None:
        self.hedef_boyut = hedef_boyut

        # Eğitim dönüşüm boru hattı (Data Augmentation Pipeline)
        self.egitim_donusumu = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=25),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.RandomErasing(p=0.4, scale=(0.02, 0.2), value="random"),
        ])

    def donustur_tensor(self, tensor_chw: torch.Tensor) -> torch.Tensor:
        """(C, H, W) veya (B, C, H, W) PyTorch tensörüne dönüşümleri uygular."""
        return self.egitim_donusumu(tensor_chw)

    def donustur_numpy(self, gorsel_hwc: np.ndarray) -> np.ndarray:
        """(H, W, C) float32 [0.0, 1.0] NumPy görseline torchvision dönüşümlerini uygular."""
        # HWC -> CHW tensöre çevir
        t = torch.from_numpy(np.transpose(gorsel_hwc, (2, 0, 1))).float()
        t_aug = self.egitim_donusumu(t)
        # CHW -> HWC NumPy'a geri çevir
        aug_hwc = np.transpose(t_aug.numpy(), (1, 2, 0))
        return np.clip(aug_hwc, 0.0, 1.0)
