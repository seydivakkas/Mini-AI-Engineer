"""Sentetik Hücresel/Medikal Doku Anlamsal Bölütleme Veri Seti Modülü.

Bu modül; Arka plan (0), Hücre Gövdesi (1) ve Hücre Çekirdeği (2) içeren
3 sınıflı sentetik mikroskobik doku görselleri ve piksel maskeleri üretir,
PyTorch Dataset ve DataLoader altyapısını kurar.
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SentetikBolutlemeDataset(Dataset):
    """Anlamsal bölütleme için sentetik görsel ve piksel maskesi üreten veri kümesi."""

    SINIFLAR = ["Arka Plan", "Hücre Gövdesi", "Çekirdek"]

    def __init__(
        self,
        ornek_sayisi: int = 40,
        img_size: int = 128,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.ornek_sayisi = ornek_sayisi
        self.img_size = img_size
        self.seed = seed
        self.veriler, self.maskeler = self._veri_uret()

    def _veri_uret(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Sentetik mikroskobik doku görselleri ve maskelerini üretir."""
        np.random.seed(self.seed)
        veriler = []
        maskeler = []

        for _ in range(self.ornek_sayisi):
            # Arka plan dokusu (Hafif pembemsi stroma dokusu)
            gorsel = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            gorsel[:, :, 0] = np.random.randint(180, 210)  # R
            gorsel[:, :, 1] = np.random.randint(150, 180)  # G
            gorsel[:, :, 2] = np.random.randint(170, 200)  # B

            # Dokusal gürültü
            gurultu = np.random.normal(0, 10, (self.img_size, self.img_size, 3)).astype(np.int16)
            gorsel = np.clip(gorsel.astype(np.int16) + gurultu, 0, 255).astype(np.uint8)

            maske = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

            # 3 ila 6 adet hücre yerleştir
            num_cells = np.random.randint(3, 7)
            for _ in range(num_cells):
                cx = np.random.randint(20, self.img_size - 20)
                cy = np.random.randint(20, self.img_size - 20)
                r_cell = np.random.randint(14, 24)
                r_nucleus = max(5, r_cell // 2)

                # 1. Hücre Gövdesi (Sitoplazma - Mavi/Mor tonlar, Maske = 1)
                cv2.circle(gorsel, (cx, cy), r_cell, (120, 90, 180), -1)
                cv2.circle(maske, (cx, cy), r_cell, 1, -1)

                # 2. Hücre Çekirdeği (Nükleus - Koyu Mor/Siyah, Maske = 2)
                cv2.circle(gorsel, (cx, cy), r_nucleus, (60, 30, 110), -1)
                cv2.circle(maske, (cx, cy), r_nucleus, 2, -1)

            veriler.append(gorsel)
            maskeler.append(maske)

        return veriler, maskeler

    def __len__(self) -> int:
        return self.ornek_sayisi

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gorsel_np = self.veriler[idx]
        maske_np = self.maskeler[idx]

        # Normalizasyon: (H, W, C) [0..255] -> (C, H, W) [0..1]
        gorsel_tensor = torch.from_numpy(gorsel_np).permute(2, 0, 1).float() / 255.0
        maske_tensor = torch.from_numpy(maske_np).long()

        return gorsel_tensor, maske_tensor


class VeriYoneticisi:
    """Veri setini Train ve Validation olarak bölen ve DataLoader sağlayan sınıf."""

    @staticmethod
    def dataloader_olustur(
        train_adet: int = 48,
        val_adet: int = 16,
        img_size: int = 128,
        batch_size: int = 8,
    ) -> Tuple[DataLoader, DataLoader, SentetikBolutlemeDataset, SentetikBolutlemeDataset]:
        """Eğitim ve doğrulama veri yükleyicilerini döndürür."""
        train_dataset = SentetikBolutlemeDataset(ornek_sayisi=train_adet, img_size=img_size, seed=42)
        val_dataset = SentetikBolutlemeDataset(ornek_sayisi=val_adet, img_size=img_size, seed=123)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, train_dataset, val_dataset
