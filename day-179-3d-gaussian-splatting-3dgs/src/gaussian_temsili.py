"""
3D Gauss Temsili ve Kovaryans Matrisi Modülü (Day 179 - FAZ 9).
Kerbl et al. (2023) 3D Gaussian Splatting (3DGS) mimarisi:
Gauss denklemi: G(x) = exp(-0.5 * (x - mu)^T * Sigma^{-1} * (x - mu))
Kovaryans ayrışımı: Sigma = R * S * S^T * R^T (Pozitif Yarı-Tanımlı Simetrik Matris)
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def kuaterniyon_to_rotasyon_matrisi(q: torch.Tensor) -> torch.Tensor:
    """
    Birim kuaterniyon [w, x, y, z] tensörünü 3x3 rotasyon matrisine (R) dönüştürür.
    q: [N, 4]
    Döner: R [N, 3, 3]
    """
    q = F.normalize(q, p=2, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.stack([
        1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)
    ], dim=-1).reshape(-1, 3, 3)

    return R


class Gaussian3D(nn.Module):
    """
    3D Gauss Nokta Kümesi Parametreleri.
    - mu: 3D Konum (x, y, z)
    - scaling: 3D Ölçek (sx, sy, sz) (exp ile pozitif kılınır)
    - rotation: 3D Dönme Kuaterniyonu (w, x, y, z)
    - opacity: Opaklık katsayısı (Sigmoid ile 0..1 aralığında)
    - sh_features: Küresel Harmonik (Spherical Harmonics) Renk Katsayıları
    """

    def __init__(self, num_gaussians: int = 100):
        super().__init__()
        self.num_gaussians = num_gaussians

        # 1. 3D Konumlar (mu)
        self.mu = nn.Parameter(torch.randn(num_gaussians, 3) * 0.5)

        # 2. Ölçekleme (log uzayında tutulur, exp ile pozitifleştirilir)
        self.log_scaling = nn.Parameter(torch.randn(num_gaussians, 3) * 0.1 - 2.0)

        # 3. Rotasyon Kuaterniyonu [w, x, y, z] (Başlangıçta birim kuaterniyon [1, 0, 0, 0])
        init_q = torch.zeros(num_gaussians, 4)
        init_q[:, 0] = 1.0
        self.rotation = nn.Parameter(init_q)

        # 4. Opaklık (logit uzayında tutulur, sigmoid ile 0..1 yapılır)
        self.opacity_logits = nn.Parameter(torch.ones(num_gaussians, 1) * 0.5)

        # 5. Renk / Küresel Harmonikler (RGB DC katsayısı)
        self.colors_dc = nn.Parameter(torch.rand(num_gaussians, 3))

    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self.log_scaling)

    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits)

    def get_rotation_matrix(self) -> torch.Tensor:
        return kuaterniyon_to_rotasyon_matrisi(self.rotation)

    def kovaryans_3d_hesapla(self) -> torch.Tensor:
        """
        Sigma = R * S * S^T * R^T
        Döner: Sigma [N, 3, 3] (Pozitif Yarı-Tanımlı Kovaryans Matrisi)
        """
        S = self.get_scaling()  # [N, 3]
        R = self.get_rotation_matrix()  # [N, 3, 3]

        # S matrisi oluştur [N, 3, 3]
        S_mat = torch.diag_embed(S)
        M = torch.bmm(R, S_mat)  # M = R * S
        Sigma = torch.bmm(M, M.transpose(1, 2))  # Sigma = M * M^T
        return Sigma
