r"""
Tesla 3D Occupancy Network ve Hacimsel Voksel Akış Çekirdeği
============================================================
Bu modül; 3 Boyutlu Hacimsel Voksel Doluluk Izgarasını ($N_x \times N_y \times N_z$),
her vokselin doluluk olasılığını ($P_{\text{occ}} \in [0, 1]$), 3D Voksel Akış
Hızını ($\vec{v} = [v_x, v_y, v_z]^T$) ve Kutuya Sığmayan (Arbitrary-Shaped)
engellerin tespitini gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class Tesla3DOccupancyNetwork:
    """
    3D Voksel Doluluk ve Voxel Flow Hacimsel Sinir Ağı.
    """
    def __init__(
        self,
        grid_dim_x: int = 50,
        grid_dim_y: int = 50,
        grid_dim_z: int = 16,
        voxel_res_xy_m: float = 1.0,
        voxel_res_z_m: float = 0.5
    ):
        self.nx = grid_dim_x
        self.ny = grid_dim_y
        self.nz = grid_dim_z
        self.res_xy = voxel_res_xy_m
        self.res_z = voxel_res_z_m

        # 3D Doluluk Logitleri (Nx x Ny x Nz)
        self.occupancy_logits = np.full((self.nx, self.ny, self.nz), -4.0, dtype=np.float32)
        
        # 3D Voksel Akış Hız Matrisi (Nx x Ny x Nz x 3) [vx, vy, vz]
        self.voxel_flow = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float32)

        # Semantik Sınıf Matrisi (Nx x Ny x Nz) (0: Boş, 1: Yol, 2: Araç, 3: Yaya, 4: Genel Engel)
        self.semantic_classes = np.zeros((self.nx, self.ny, self.nz), dtype=np.uint8)

    def insert_synthetic_scene(self):
        """
        Sahneye Zemin, Öncü Araç (Hareketli), Yaya ve Devrilmiş Ağaç (Kutulanamaz Engel) ekler.
        """
        cx, cy = self.nx // 2, self.ny // 2

        # 1. Statik Yol Yüzeyi (Z = 0, 1. katman)
        self.occupancy_logits[:, :, 0:2] = 4.0
        self.semantic_classes[:, :, 0:2] = 1

        # 2. Öncü Araç (Önde X = +15m, Hız = [15, 0, 0] m/s)
        car_gx = cx + 15
        car_gy = cy
        self.occupancy_logits[car_gx-2 : car_gx+2, car_gy-1 : car_gy+1, 2:5] = 5.0
        self.voxel_flow[car_gx-2 : car_gx+2, car_gy-1 : car_gy+1, 2:5, 0] = 15.0  # Vx = 15 m/s
        self.semantic_classes[car_gx-2 : car_gx+2, car_gy-1 : car_gy+1, 2:5] = 2

        # 3. Yaya (Sağda Y = +6m, Hız = [0, -1.2, 0] m/s)
        ped_gx = cx + 5
        ped_gy = cy + 6
        self.occupancy_logits[ped_gx, ped_gy, 2:5] = 4.5
        self.voxel_flow[ped_gx, ped_gy, 2:5, 1] = -1.2  # Vy = -1.2 m/s
        self.semantic_classes[ped_gx, ped_gy, 2:5] = 3

        # 4. Yola Devrilmiş Ağaç Gövdesi (Kutuya Sığmayan Düzensiz Engel - Arbitrary Shape)
        tree_gx = min(cx + 20, self.nx - 1)
        for dy in range(-4, 5):
            self.occupancy_logits[tree_gx, cy + dy, 2] = 4.0
            self.semantic_classes[tree_gx, cy + dy, 2] = 4

    def compute_occupancy_probabilities(self, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sigmoid ile Doluluk Olasılıklarını hesaplar ve eşikler: P_occ = 1 / (1 + exp(-z)).
        """
        probs = 1.0 / (1.0 + np.exp(-self.occupancy_logits))
        binary_mask = probs >= threshold
        return probs, binary_mask

    def query_point_velocity(self, x_m: float, y_m: float, z_m: float) -> Tuple[float, np.ndarray]:
        """
        Belirli bir metrik 3D konumdaki doluluk olasılığını ve 3D akış hızını sorgular.
        """
        cx, cy = self.nx // 2, self.ny // 2
        gx = int(cx + (x_m / self.res_xy))
        gy = int(cy + (y_m / self.res_xy))
        gz = int(z_m / self.res_z)

        if 0 <= gx < self.nx and 0 <= gy < self.ny and 0 <= gz < self.nz:
            prob = float(1.0 / (1.0 + np.exp(-self.occupancy_logits[gx, gy, gz])))
            flow = self.voxel_flow[gx, gy, gz].copy()
            return prob, flow
        return 0.0, np.zeros(3, dtype=np.float32)
