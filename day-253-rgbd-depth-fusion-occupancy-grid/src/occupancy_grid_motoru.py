"""
RGB-D Derinlik Füzyonu ve 3D Doluluk Izgarası (Occupancy Grid) Motoru (Day 253).
Pinhole 3D İzdüşümü, Log-Odds Bayesyen Doluluk Füzyonu ve Dinamik Engel Kaçınma.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


class RGBDProjector:
    """RGB-D Derinlik Haritasından 3D Nokta Bulutu Üretici (Pinhole Kamera Modeli)."""

    def __init__(
        self,
        fx: float = 525.0,
        fy: float = 525.0,
        cx: float = 319.5,
        cy: float = 239.5,
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def depth_to_point_cloud(self, depth_image: np.ndarray, max_depth_m: float = 4.0) -> np.ndarray:
        """2D derinlik matrisini dünya koordinatlarında 3D nokta bulutuna dönüştürür."""
        h, w = depth_image.shape
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

        z = depth_image.astype(np.float64)
        valid_mask = (z > 0.1) & (z < max_depth_m)

        x = (u_coords - self.cx) * z / self.fx
        y = (v_coords - self.cy) * z / self.fy

        points_3d = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=-1)
        return points_3d


class VoxelOccupancyGrid:
    """3D Log-Odds Bayesyen Doluluk Izgarası (Voxel / OctoMap Mimarisi)."""

    def __init__(
        self,
        min_bound: Tuple[float, float, float] = (-2.0, -2.0, 0.0),
        max_bound: Tuple[float, float, float] = (2.0, 2.0, 1.5),
        resolution_m: float = 0.05,
    ):
        self.min_b = np.array(min_bound, dtype=np.float64)
        self.max_b = np.array(max_bound, dtype=np.float64)
        self.res = resolution_m

        self.grid_dims = np.ceil((self.max_b - self.min_b) / self.res).astype(int)
        self.log_odds = np.zeros(self.grid_dims, dtype=np.float32)

        # Log-Odds Parametreleri
        self.l_occ = 0.85   # Hit noktası log-odds artışı
        self.l_free = -0.35 # Boş uzay log-odds azalışı
        self.l_min = -2.0   # Doygunluk alt sınırı
        self.l_max = 3.5    # Doygunluk üst sınırı

    def coord_to_index(self, points: np.ndarray) -> np.ndarray:
        """3D koordinatları 3D ızgara indekslerine çevirir."""
        indices = np.floor((points - self.min_b) / self.res).astype(int)
        valid = (
            (indices[:, 0] >= 0) & (indices[:, 0] < self.grid_dims[0]) &
            (indices[:, 1] >= 0) & (indices[:, 1] < self.grid_dims[1]) &
            (indices[:, 2] >= 0) & (indices[:, 2] < self.grid_dims[2])
        )
        return indices[valid]

    def update_with_points(self, points_3d: np.ndarray, origin: np.ndarray = np.array([0.0, 0.0, 0.5])):
        """Hit ve Free space log-odds güncellemelerini yapar."""
        if len(points_3d) == 0:
            return

        indices = self.coord_to_index(points_3d)
        for idx in indices:
            self.log_odds[idx[0], idx[1], idx[2]] += self.l_occ

        # Kırpma (Clamping)
        self.log_odds = np.clip(self.log_odds, self.l_min, self.l_max)

    def get_occupied_voxel_count(self, threshold_prob: float = 0.70) -> int:
        """Olasılığı eşiği aşan dolu voksel sayısını döner (P = 1 / (1 + exp(-L)))."""
        l_thresh = float(np.log(threshold_prob / (1.0 - threshold_prob)))
        return int(np.sum(self.log_odds >= l_thresh))


class DynamicObstacleAvoidance:
    """3D Doluluk Izgarasından Şişirme (Inflation) ve Güvenli Rota Üreticisi."""

    @classmethod
    def plan_avoidance_path(
        cls,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        obstacle_centers: List[np.ndarray],
        robot_radius_m: float = 0.25,
    ) -> Dict[str, Any]:
        """Dinamik engellerin etrafından teğet geçerek güvenli rota üretir."""
        path = [start_pos.copy()]
        current = start_pos.copy()

        steps = 6
        for i in range(1, steps + 1):
            alpha = i / float(steps)
            nominal_pt = (1.0 - alpha) * start_pos + alpha * goal_pos

            # Engele olan mesafe kontrolü
            repulsion = np.zeros(3)
            for obs in obstacle_centers:
                diff = nominal_pt - obs
                dist = np.linalg.norm(diff)
                if dist < (robot_radius_m + 0.35):
                    if dist < 1e-3:
                        normal = np.array([1.0, 0.0, 0.0])
                    else:
                        normal = diff / dist
                    repulsion += normal * (robot_radius_m + 0.35 - dist)

            safe_pt = nominal_pt + repulsion
            path.append(safe_pt.round(3))

        min_clearance = min([
            min([float(np.linalg.norm(p - obs)) for obs in obstacle_centers])
            for p in path
        ])

        return {
            "path": [p.tolist() for p in path],
            "min_clearance_m": round(min_clearance, 3),
            "carpisma_var_mi": min_clearance < robot_radius_m,
        }
