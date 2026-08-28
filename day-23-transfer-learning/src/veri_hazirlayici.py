"""Transfer Öğrenme Veri Seti ve ImageNet Ön İşleme Modülü.

Bu modül; önceden eğitilmiş omurgaların (ResNet/EfficientNet) beklediği ImageNet
renk ve uzamsal istatistiklerine (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
uygun veri seti ve DataLoader üretimini sağlar.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset


class ImageNetDataset(Dataset):
    """ImageNet istatistikleriyle z-score normalize edilmiş PyTorch veri seti."""

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __init__(self, gorseller_hwc: np.ndarray, etiketler: np.ndarray) -> None:
        self.gorseller = gorseller_hwc.astype(np.float32)
        self.etiketler = etiketler.astype(np.int64)

    def __len__(self) -> int:
        return len(self.gorseller)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.gorseller[idx]  # (H, W, 3) [0.0, 1.0]
        chw = np.transpose(img, (2, 0, 1))
        norm_chw = (chw - self.IMAGENET_MEAN) / self.IMAGENET_STD
        return torch.from_numpy(norm_chw).float(), torch.tensor(self.etiketler[idx], dtype=torch.long)


class TransferVeriYoneticisi:
    """Veri üretimi, bölümleme ve ImageNet uyumlu DataLoader yönetim sınıfı."""

    def __init__(self, hedef_boyut: Tuple[int, int] = (64, 64), random_state: int = 42) -> None:
        self.hedef_boyut = hedef_boyut
        self.random_state = random_state
        np.random.seed(random_state)

    def sentetik_veri_seti_uret(
        self, sinif_basina_ornek: int = 40
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """4 sınıflı sentetik görsel veri seti üretir."""
        H, W = self.hedef_boyut
        siniflar = ["Vazo", "Kumaş", "Rozet", "Ahşap"]
        X_list, y_list = [], []

        for sinif_idx, sinif_adi in enumerate(siniflar):
            for i in range(sinif_basina_ornek):
                img = np.zeros((H, W, 3), dtype=np.float32)
                if sinif_adi == "Vazo":
                    for y in range(H):
                        img[y, :, 0] = 0.1 + 0.6 * (y / H)
                        img[y, :, 1] = 0.3 + 0.4 * (y / H)
                        img[y, :, 2] = 0.8
                    cv2.ellipse(img, (W // 2, H // 2), (W // 4, H // 3), 0, 0, 360, (0.9, 0.9, 0.2), -1)
                elif sinif_adi == "Kumaş":
                    img[:, :, 2] = 0.6 + 0.3 * np.sin(np.linspace(0, 12 * np.pi, W))
                    img[:, :, 0] = 0.2
                    img[:, :, 1] = 0.2
                    for x in range(0, W, 6):
                        cv2.line(img, (x, 0), (x, H), (0.1, 0.8, 0.8), 1)
                elif sinif_adi == "Rozet":
                    img[:, :] = (0.1, 0.1, 0.1)
                    cv2.circle(img, (W // 2, H // 2), W // 3, (0.1, 0.8, 0.9), 3)
                    cv2.circle(img, (W // 2, H // 2), W // 5, (0.2, 0.9, 1.0), -1)
                elif sinif_adi == "Ahşap":
                    for y in range(H):
                        c = 0.2 + 0.5 * (np.sin(y * 0.4) ** 2)
                        img[y, :, 0] = c * 0.3
                        img[y, :, 1] = c * 0.6
                        img[y, :, 2] = c * 0.9

                gurultu = np.random.normal(0, 0.03, (H, W, 3)).astype(np.float32)
                img = np.clip(img + gurultu, 0.0, 1.0)
                X_list.append(img)
                y_list.append(sinif_idx)

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64), siniflar

    def veri_bol_ve_yukleyicileri_olustur(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_orani: float = 0.15,
        test_orani: float = 0.20,
        batch_size: int = 16,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
        """Stratified bölme uygular ve ImageNetDataset DataLoader'larını oluşturur."""
        test_toplam = val_orani + test_orani
        _, counts = np.unique(y, return_counts=True)
        stratify_ilk = y if np.min(counts) >= 4 else None

        X_train, X_gecici, y_train, y_gecici = train_test_split(
            X, y, test_size=test_toplam, stratify=stratify_ilk, random_state=self.random_state
        )

        val_orani_gecici = val_orani / test_toplam
        _, gecici_counts = np.unique(y_gecici, return_counts=True)
        stratify_ikinci = y_gecici if np.min(gecici_counts) >= 2 else None

        X_val, X_test, y_val, y_test = train_test_split(
            X_gecici, y_gecici, test_size=1.0 - val_orani_gecici,
            stratify=stratify_ikinci, random_state=self.random_state
        )

        train_loader = DataLoader(ImageNetDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ImageNetDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(ImageNetDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, X_test, y_test
