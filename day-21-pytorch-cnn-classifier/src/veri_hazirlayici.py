"""PyTorch Veri Seti ve Veri Yükleyici (Dataset & DataLoader) Yönetimi.

Bu modül; PyTorch torch.utils.data.Dataset sınıfından türetilen SentetikGorselDataset
ile çok sınıflı görsel veri setini yükler, tensör dönüşümlerini (HWC -> CHW) uygular
ve Stratified DataLoader nesnelerini hazırlar.
"""

from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset


class SentetikGorselDataset(Dataset):
    """PyTorch özel görsel veri seti sınıfı."""

    def __init__(
        self,
        gorseller: np.ndarray,
        etiketler: np.ndarray,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        """SentetikGorselDataset'i ilklendirir.

        Args:
            gorseller: (N, H, W, C) float32 [0.0, 1.0] NumPy dizisi.
            etiketler: (N,) int64 etiket dizisi.
            mean: Kanal bazında ortalama normalizasyon değerleri.
            std: Kanal bazında standart sapma değerleri.
        """
        self.gorseller = gorseller.astype(np.float32)
        self.etiketler = etiketler.astype(np.int64)
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        """Veri setindeki toplam örnek sayısı."""
        return len(self.gorseller)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Belirtilen indeksteki örneği tensör formatında döndürür.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - gorsel_tensor: (C, H, W) normalize edilmiş float32 tensör.
                - etiket_tensor: skaler int64 tensör.
        """
        gorsel_hwc = self.gorseller[idx]  # (H, W, C) [0.0, 1.0]
        # HWC -> CHW dönüşümü
        gorsel_chw = np.transpose(gorsel_hwc, (2, 0, 1))

        # Standart z-score normalizasyonu: (x - mean) / std
        gorsel_norm = (gorsel_chw - self.mean) / self.std

        tensor_x = torch.from_numpy(gorsel_norm).float()
        tensor_y = torch.tensor(self.etiketler[idx], dtype=torch.long)
        return tensor_x, tensor_y


class VeriYoneticisi:
    """Sentetik görsel üretimi, veri bölümleme ve DataLoader yönetim sınıfı."""

    def __init__(
        self,
        hedef_boyut: Tuple[int, int] = (64, 64),
        random_state: int = 42,
    ) -> None:
        self.hedef_boyut = hedef_boyut
        self.random_state = random_state
        np.random.seed(random_state)

    def sentetik_veri_seti_uret(
        self, sinif_basina_ornek: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Farklı geometrik ve spektral dokularda 4 sınıflı görsel veri seti üretir."""
        H, W = self.hedef_boyut
        siniflar = ["Vazo", "Kumaş", "Rozet", "Ahşap"]
        X_listesi = []
        y_listesi = []

        for sinif_idx, sinif_adi in enumerate(siniflar):
            for i in range(sinif_basina_ornek):
                img = np.zeros((H, W, 3), dtype=np.float32)

                if sinif_adi == "Vazo":
                    # Mavi-Camgöbeği degrade ve elips
                    for y in range(H):
                        img[y, :, 0] = 0.1 + 0.6 * (y / H)
                        img[y, :, 1] = 0.3 + 0.4 * (y / H)
                        img[y, :, 2] = 0.8
                    cv2.ellipse(img, (W // 2, H // 2), (W // 4, H // 3), 0, 0, 360, (0.9, 0.9, 0.2), -1)

                elif sinif_adi == "Kumaş":
                    # Kırmızı/Bordo ızgara dokusu
                    img[:, :, 2] = 0.6 + 0.3 * np.sin(np.linspace(0, 12 * np.pi, W))
                    img[:, :, 0] = 0.2
                    img[:, :, 1] = 0.2
                    for x in range(0, W, 6):
                        cv2.line(img, (x, 0), (x, H), (0.1, 0.8, 0.8), 1)

                elif sinif_adi == "Rozet":
                    # Altın/Sarı dairesel halkalar ve yıldız deseni
                    img[:, :] = (0.1, 0.1, 0.1)
                    cv2.circle(img, (W // 2, H // 2), W // 3, (0.1, 0.8, 0.9), 3)
                    cv2.circle(img, (W // 2, H // 2), W // 5, (0.2, 0.9, 1.0), -1)

                elif sinif_adi == "Ahşap":
                    # Kahverengi/Yeşil yatay lifler ve halkalar
                    for y in range(H):
                        c = 0.2 + 0.5 * (np.sin(y * 0.4) ** 2)
                        img[y, :, 0] = c * 0.3
                        img[y, :, 1] = c * 0.6
                        img[y, :, 2] = c * 0.9

                # Hafif gürültü ekle
                gurultu = np.random.normal(0, 0.03, (H, W, 3)).astype(np.float32)
                img = np.clip(img + gurultu, 0.0, 1.0)

                X_listesi.append(img)
                y_listesi.append(sinif_idx)

        X = np.array(X_listesi, dtype=np.float32)
        y = np.array(y_listesi, dtype=np.int64)
        return X, y, siniflar

    def veri_bol_ve_yukleyicileri_olustur(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_orani: float = 0.15,
        test_orani: float = 0.15,
        batch_size: int = 16,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
        """Veriyi Stratified böler ve PyTorch Train/Val/Test DataLoader'larını oluşturur."""
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

        train_ds = SentetikGorselDataset(X_train, y_train)
        val_ds = SentetikGorselDataset(X_val, y_val)
        test_ds = SentetikGorselDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, X_test, y_test
