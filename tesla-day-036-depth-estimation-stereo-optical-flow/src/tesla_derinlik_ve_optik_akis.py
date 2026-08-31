"""
Tesla Derinlik Tahmini ve Geometrik Optik Akış Çekirdeği
=========================================================
Bu modül; Stereo Disparity derinlik dönüşümünü ($Z = fB/d$), derinlik belirsizliğini,
Lucas-Kanade optik akışını ve Optik Genişleme ile Çarpışma Süresi (Time-To-Contact - TTC)
kestirimini gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaDepthAndOpticalFlowEstimator:
    """
    Tesla FSD Derinlik ve Optik Akış Motoru.
    """
    def __init__(self, focal_length_px: float = 1200.0, baseline_m: float = 0.50):
        self.f_px = focal_length_px
        self.b_m = baseline_m

    def disparity_to_depth(self, disparity_px: np.ndarray, min_disp: float = 0.5) -> np.ndarray:
        """Z = (f * B) / d."""
        disp_clipped = np.clip(disparity_px, min_disp, 250.0)
        return (self.f_px * self.b_m) / disp_clipped

    def compute_depth_uncertainty(self, depth_m: np.ndarray, sigma_disp_px: float = 0.5) -> np.ndarray:
        """sigma_Z = (Z^2 / (f * B)) * sigma_d."""
        return (depth_m ** 2 / (self.f_px * self.b_m)) * sigma_disp_px

    def compute_time_to_contact(self, depth_m: float, rel_speed_mps: float) -> float:
        """TTC = Z / v_rel (Saniye)."""
        if rel_speed_mps <= 0.0:  # Uzaklaşıyor veya duruyor
            return 999.0
        return float(depth_m / rel_speed_mps)

    def estimate_lucas_kanade_flow(
        self,
        patch_i1: np.ndarray,
        patch_i2: np.ndarray
    ) -> Tuple[float, float]:
        """
        2D Lucas-Kanade Optik Akış Hız Vektörü: [u_t, v_t]^T = (A^T A)^-1 A^T b.
        """
        assert patch_i1.shape == patch_i2.shape
        # Gradyanlar: Ix, Iy, It
        Ix = np.gradient(patch_i1, axis=1).flatten()
        Iy = np.gradient(patch_i1, axis=0).flatten()
        It = (patch_i2 - patch_i1).flatten()

        A = np.column_stack((Ix, Iy))
        b = -It

        # Normal denklemler: (A^T A) v = A^T b
        ATA = A.T @ A
        if np.linalg.det(ATA) < 1e-4:
            return 0.0, 0.0  # Doku yetersiz (Aperture problem)

        velocity = np.linalg.inv(ATA) @ A.T @ b
        return float(velocity[0]), float(velocity[1])

    def block_matching_disparity_1d(
        self,
        left_scanline: np.ndarray,
        right_scanline: np.ndarray,
        max_disp: int = 64,
        window_size: int = 5
    ) -> np.ndarray:
        """
        1D Epipolar çizgi üzerinde SAD (Sum of Absolute Differences) blok eşleme.
        """
        w = len(left_scanline)
        disparities = np.zeros(w, dtype=np.float64)
        half_w = window_size // 2

        for u in range(half_w + max_disp, w - half_w):
            left_block = left_scanline[u - half_w : u + half_w + 1]
            best_sad = float('inf')
            best_d = 0

            for d in range(0, max_disp):
                right_u = u - d
                right_block = right_scanline[right_u - half_w : right_u + half_w + 1]
                sad = float(np.sum(np.abs(left_block - right_block)))
                if sad < best_sad:
                    best_sad = sad
                    best_d = d

            disparities[u] = best_d

        return disparities
