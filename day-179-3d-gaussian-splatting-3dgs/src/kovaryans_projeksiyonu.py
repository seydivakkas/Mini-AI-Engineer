"""
3D'den 2D Ekran Düzlemine Kovaryans Projeksiyonu Modülü (Day 179 - FAZ 9).
Zwicker et al. (2001) EWA Splatting ve Kerbl et al. (2023) 3DGS Formülasyonu:
Sigma' = J * W * Sigma * W^T * J^T + 0.3 * I_2x2 (Anti-Aliasing Düşük Geçiren Filtre)
"""

from typing import Tuple
import torch
import torch.nn as nn


class KovaryansProjeksiyonu:
    """3D Gauss Kovaryansını 2D Kamera Ekranına Yansıtan Jacobian Projeksiyon Motoru."""

    @classmethod
    def jacobian_hesapla(
        cls,
        pts_cam: torch.Tensor,
        fx: float = 500.0,
        fy: float = 500.0,
    ) -> torch.Tensor:
        """
        Kamera koordinatlarındaki noktalar [N, 3] için izdüşüm Jacobian matrisini (J) hesaplar.
        J = [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]  [N, 2, 3]
        """
        x = pts_cam[:, 0]
        y = pts_cam[:, 1]
        z = torch.clamp(pts_cam[:, 2], min=0.1)  # Sıfıra bölmeyi engelle

        N = pts_cam.shape[0]
        J = torch.zeros(N, 2, 3, device=pts_cam.device)

        J[:, 0, 0] = fx / z
        J[:, 0, 2] = - (fx * x) / (z ** 2)
        J[:, 1, 1] = fy / z
        J[:, 1, 2] = - (fy * y) / (z ** 2)

        return J

    @classmethod
    def izdusum_2d_kovaryans(
        cls,
        mu_3d: torch.Tensor,
        sigma_3d: torch.Tensor,
        view_matrix_R: torch.Tensor,
        view_matrix_T: torch.Tensor,
        fx: float = 500.0,
        fy: float = 500.0,
        cx: float = 256.0,
        cy: float = 256.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        3D Gauss'ları 2D ekran koordinatlarına ve 2D kovaryans matrislerine dönüştürür.
        Döner: (mu_2d [N, 2], sigma_2d [N, 2, 2])
        """
        # 1. Kamera Koordinatlarına Dönüşüm
        # pts_cam = mu_3d @ R^T + T
        pts_cam = torch.matmul(mu_3d, view_matrix_R.t()) + view_matrix_T

        # 2. 2D Ekran Merkezi (Piksel Koordinatları)
        z = torch.clamp(pts_cam[:, 2], min=0.1)
        u = (pts_cam[:, 0] * fx / z) + cx
        v = (pts_cam[:, 1] * fy / z) + cy
        mu_2d = torch.stack([u, v], dim=-1)

        # 3. Jacobian Matrisi J [N, 2, 3]
        J = cls.jacobian_hesapla(pts_cam, fx=fx, fy=fy)

        # 4. W Matrisi (Kamera rotasyonu) [N, 3, 3]
        N = mu_3d.shape[0]
        W = view_matrix_R.unsqueeze(0).expand(N, 3, 3)

        # 5. Sigma_cam = W * Sigma_3d * W^T
        sigma_cam = torch.bmm(torch.bmm(W, sigma_3d), W.transpose(1, 2))

        # 6. Sigma_2d = J * Sigma_cam * J^T + 0.3 * I
        sigma_2d = torch.bmm(torch.bmm(J, sigma_cam), J.transpose(1, 2))

        # Anti-aliasing ve sayısal kararlılık için köşegenlere +0.3 ekle
        sigma_2d[:, 0, 0] += 0.3
        sigma_2d[:, 1, 1] += 0.3

        return mu_2d, sigma_2d
