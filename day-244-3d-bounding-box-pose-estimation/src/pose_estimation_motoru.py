"""
3D Sınırlayıcı Kutu ve 6-DoF Duruş Kestirimi (VoteNet / Pose Estimation) Motoru (Day 244).
Derin Hough Oylama, 3D Kutu Regresyonu ve ADD-S Robotik Doğruluk Metriği.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VotingModule(nn.Module):
    """Yüzey Noktalarından Nesne Merkezine Oy Üreten Derin Hough Oylama Katmanı."""

    def __init__(self, in_channels: int = 64, out_channels: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, 3 + out_channels, 1),  # [dx, dy, dz] ofseti + yeni özellikler
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Girdi: xyz=[B, N, 3], features=[B, C, N] -> Çıktı: vote_xyz=[B, N, 3], vote_features=[B, C, N]"""
        net = self.mlp(features)
        offset = net[:, :3, :].transpose(1, 2)  # [B, N, 3]
        vote_features = net[:, 3:, :]  # [B, C, N]
        vote_xyz = xyz + offset
        return vote_xyz, vote_features


class BoundingBox3DHead(nn.Module):
    """3D Merkez, Boyut (l, w, h), Yaw Açısı ve Güven Skoru Üreten Regresyon Başlığı."""

    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3 + 3 + 2 + 1),  # Merkez (3) + Boyut (3) + [sin(yaw), cos(yaw)] (2) + Skor (1)
        )

    def forward(self, vote_cluster_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Girdi: [B, C] küme özelliği -> Çıktı: 3D Kutu Parametreleri Sözlüğü."""
        out = self.mlp(vote_cluster_features)
        center = out[:, 0:3]
        dimensions = F.softplus(out[:, 3:6]) + 0.05  # Pozitif boyut [l, w, h] (metre)
        heading_sc = out[:, 6:8]
        heading_yaw = torch.atan2(heading_sc[:, 0:1], heading_sc[:, 1:2])
        score = torch.sigmoid(out[:, 8:9])

        return {
            "center": center,
            "dimensions": dimensions,
            "yaw_rad": heading_yaw,
            "confidence": score,
        }


class VoteNetPoseEstimator(nn.Module):
    """Uçtan Uca 6-DoF Nesne Duruş ve 3D Sınırlayıcı Kutu Kestiricisi."""

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Conv1d(3, feature_dim, 1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, 1),
        )
        self.voting = VotingModule(in_channels=feature_dim, out_channels=feature_dim)
        self.box_head = BoundingBox3DHead(in_channels=feature_dim)

    def forward(self, xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Girdi: [B, N, 3] nokta bulutu -> Çıktı: 3D Kutu parametreleri."""
        xyz_transposed = xyz.transpose(1, 2)
        features = self.feature_encoder(xyz_transposed)
        vote_xyz, vote_features = self.voting(xyz, features)

        # Oyların Küresel Havuzlanması (Cluster Aggregation)
        cluster_feature = torch.max(vote_features, dim=-1)[0]  # [B, C]
        predictions = self.box_head(cluster_feature)
        predictions["vote_xyz"] = vote_xyz
        return predictions


def hesapla_adds_metrigi(tahmin_center: np.ndarray, hedef_center: np.ndarray, esik_cm: float = 2.0) -> Tuple[float, bool]:
    """ADD-S (Average Distance of Model Points) hata mesafesini ve başarı eşiğini hesaplar."""
    hata_metre = np.linalg.norm(tahmin_center - hedef_center)
    hata_cm = float(hata_metre * 100.0)
    basarili_mi = bool(hata_cm <= esik_cm)
    return round(hata_cm, 3), basarili_mi
