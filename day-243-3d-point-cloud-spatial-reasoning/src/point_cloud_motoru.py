"""
3D Nokta Bulutu ve Mekansal Akıl Yürütme (PointNet++) Motoru (Day 243).
Farthest Point Sampling (FPS), Ball Query, Hiyerarşik Set Abstraction ve Tutma Afordansı.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def farthest_point_sampling(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """En Uzak Nokta Örneklemesi (FPS): [B, N, 3] -> [B, npoint] indis tensörü."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def ball_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """Küresel Komşuluk Sorgusu: Her merkez için yarıçap içindeki nsample noktayı toplar."""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, -1)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class PointNetSetAbstraction(nn.Module):
    """Hiyerarşik Nokta Kümesi Soyutlama Katmanı (Set Abstraction)."""

    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, mlp: List[int]):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        layers = []
        last_channel = in_channel + 3  # Bağıl koordinat (dx, dy, dz) eklenir
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Girdi: xyz=[B, N, 3], points=[B, in_channel, N] -> Çıktı: new_xyz=[B, S, 3], new_points=[B, out_channel, S]"""
        B, N, C = xyz.shape
        fps_idx = farthest_point_sampling(xyz, self.npoint)
        new_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).repeat(1, 1, 3))

        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = torch.gather(xyz.unsqueeze(1).repeat(1, self.npoint, 1, 1), 2, idx.unsqueeze(-1).repeat(1, 1, 1, 3))
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

        if points is not None:
            grouped_points = torch.gather(points.transpose(1, 2).unsqueeze(1).repeat(1, self.npoint, 1, 1), 2, idx.unsqueeze(-1).repeat(1, 1, 1, points.shape[1]))
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm

        new_points = new_points.permute(0, 3, 2, 1)  # [B, in_channel+3, nsample, npoint]
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, 2)[0]  # Max-pooling: [B, out_channel, npoint]

        return new_xyz, new_points


class PointNetPlusPlusModel(nn.Module):
    """3D Mekansal Akıl Yürütme ve Robotik Tutma Afordansı (Grasp Affordance) Modeli."""

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=16, in_channel=0, mlp=[32, 64])
        self.sa2 = PointNetSetAbstraction(npoint=32, radius=0.4, nsample=8, in_channel=64, mlp=[64, 128])

        self.fc_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """[B, N, 3] boyutlu nokta bulutundan küresel özellik ve tutma puanı üretir."""
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        kuresel_vektor = torch.max(l2_points, 2)[0]
        skor = self.fc_head(kuresel_vektor)
        return skor


def ornek_3d_fincan_bulutu_olustur(nokta_sayisi: int = 512) -> np.ndarray:
    """Robotik test için kulplu 3D fincan nokta bulutu sentetik geometrisi."""
    np.random.seed(42)
    govde_noktalari = int(nokta_sayisi * 0.7)
    kulp_noktalari = nokta_sayisi - govde_noktalari

    # Silindirik Gövde
    theta = np.random.uniform(0, 2 * np.pi, govde_noktalari)
    z_govde = np.random.uniform(-0.5, 0.5, govde_noktalari)
    r_govde = 0.3
    x_govde = r_govde * np.cos(theta)
    y_govde = r_govde * np.sin(theta)
    govde = np.stack([x_govde, y_govde, z_govde], axis=-1)

    # Torus Kulp Geometrisi
    phi = np.random.uniform(-np.pi / 2, np.pi / 2, kulp_noktalari)
    x_kulp = 0.3 + 0.15 * np.cos(phi)
    y_kulp = np.zeros(kulp_noktalari)
    z_kulp = 0.2 * np.sin(phi)
    kulp = np.stack([x_kulp, y_kulp, z_kulp], axis=-1)

    bulut = np.concatenate([govde, kulp], axis=0)
    return bulut.astype(np.float32)
