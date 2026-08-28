"""
Görsel Yeniden Tanımlama (Re-Identification / Re-ID) Öznitelik Çıkarıcı.
Görsel kırpıntısını (Crop) 128 Boyutlu L2 Normalize Birim Küreye Yansıtır.
"""

from typing import List
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReIDCnnAg(nn.Module):
    """Hafif 128D Re-ID Gömmesi Çıkarıcı Evrişimli Sinir Ağı."""

    def __init__(self, cikti_boyutu: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, cikti_boyutu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat = torch.flatten(feat, 1)
        emb = self.fc(feat)
        # L2 Normalizasyonu (Birim Küre)
        emb_norm = F.normalize(emb, p=2, dim=1)
        return emb_norm


class ReIDEmbeddingCikarici:
    """Görsellerden kırpılan nesneler için Re-ID gömmelerini yönetir."""

    def __init__(self, feature_dim: int = 128, device: str = "cpu"):
        self.feature_dim = feature_dim
        self.device = torch.device(device)
        self.model = ReIDCnnAg(cikti_boyutu=feature_dim).to(self.device)
        self.model.eval()

    def cikar(self, kirpintilar: List[np.ndarray]) -> np.ndarray:
        """
        N adet RGB görsel kırpıntısını (Cropped BBoxes) alır ve
        (N, 128) boyutunda L2 normalize Re-ID matrisi döndürür.
        """
        if not kirpintilar:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        batch_tensors = []
        for crop in kirpintilar:
            if crop is None or crop.size == 0:
                crop = np.zeros((64, 32, 3), dtype=np.uint8)
            # Standart Re-ID girdi boyutu (64x32)
            resized = cv2.resize(crop, (32, 64))
            tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
            batch_tensors.append(tensor)

        batch = torch.stack(batch_tensors).to(self.device)
        with torch.no_grad():
            embeddings = self.model(batch).cpu().numpy()

        return embeddings.astype(np.float32)

    @staticmethod
    def kosinus_mesafesi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        İki L2 normalize embedding kümesi arasındaki Kosinüs mesafesini hesaplar:
        d(a, b) = 1 - dot(a, b)
        """
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-6)
        sim = np.dot(a_norm, b_norm.T)
        return np.clip(1.0 - sim, 0.0, 2.0)
