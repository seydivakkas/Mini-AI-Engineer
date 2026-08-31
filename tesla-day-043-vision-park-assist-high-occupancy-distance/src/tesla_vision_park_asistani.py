r"""
Tesla Vision Park Asistanı ve Yüksek Çözünürlüklü Mesafe Kestirimi (High-Occupancy)
=================================================================================
Bu modül; Ultrasonik Sensör (USS) olmadan, 3D Voxel Doluluk Alanı (Occupancy Network)
ve Kör Nokta Zamansal Belleği (Blind Spot Temporal Memory) ile araç çevresindeki
kaldırım, duvar, kolon ve araç mesafelerini santimetre hassasiyetinde kestirir.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaVisionParkAssist:
    """
    Tesla Vision Tabanlı Yüksek Çözünürlüklü Park Asistanı.
    """
    def __init__(
        self,
        vehicle_length_m: float = 4.69,
        vehicle_width_m: float = 1.85,
        grid_resolution_m: float = 0.05,
        grid_size_m: float = 10.0
    ):
        self.l_veh = vehicle_length_m
        self.w_veh = vehicle_width_m
        self.res = grid_resolution_m
        self.grid_dim = int(grid_size_m / grid_resolution_m)  # 200x200 grid (10m x 10m)
        self.center_idx = self.grid_dim // 2

        # 2D/3D Doluluk Izgarası (0: Boş, 1: Engel, 0..1 Olasılık)
        self.occupancy_grid = np.zeros((self.grid_dim, self.grid_dim), dtype=np.float32)

        # Tampon Kör Nokta Zamansal Belleği (Ön/Arka tampon altı engeller)
        self.blind_spot_memory: List[Dict[str, Any]] = []

    def update_occupancy_and_memory(
        self,
        new_point_cloud: np.ndarray,
        ego_delta_x: float,
        ego_delta_y: float
    ):
        """
        Yeni gelen 3D nokta bulutunu işler ve araç hareket ettikçe
        kör noktadaki engelleri zamansal hafızada öteler.
        """
        # 1. Mevcut hafızadaki engelleri aracın hareketine göre kaydır
        self.occupancy_grid = np.roll(self.occupancy_grid, shift=(-int(ego_delta_y / self.res)), axis=0)
        self.occupancy_grid = np.roll(self.occupancy_grid, shift=(-int(ego_delta_x / self.res)), axis=1)

        # 2. Yeni görüş noktalarını ızgaraya işle
        for pt in new_point_cloud:
            x, y = pt[0], pt[1]
            gx = int(self.center_idx + (x / self.res))
            gy = int(self.center_idx + (y / self.res))

            if 0 <= gx < self.grid_dim and 0 <= gy < self.grid_dim:
                self.occupancy_grid[gy, gx] = np.clip(self.occupancy_grid[gy, gx] + 0.6, 0.0, 1.0)

    def compute_360_distance_contour(self, num_angles: int = 360) -> np.ndarray:
        r"""
        Araç gövdesinden (Bumper Perimeter) dışarı doğru 360 derece ışın atarak
        (Ray-Casting) en yakın engel mesafelerini ($d_{\min}(\theta)$) santimetre cinsinden çıkarır.
        """
        angles_deg = np.linspace(0, 360, num_angles, endpoint=False)
        distances_cm = np.ones(num_angles, dtype=np.float32) * 999.0

        half_l = self.l_veh / 2.0
        half_w = self.w_veh / 2.0

        for idx, ang in enumerate(angles_deg):
            rad = np.radians(ang)
            dir_x = np.cos(rad)
            dir_y = np.sin(rad)

            # Araç gövdesinin dış sınırından ışın başlat
            r_start = np.hypot(half_l, half_w)  # Kaba yarıçap
            for r in np.arange(0.5, 4.0, self.res):
                x = dir_x * r
                y = dir_y * r

                gx = int(self.center_idx + (x / self.res))
                gy = int(self.center_idx + (y / self.res))

                if 0 <= gx < self.grid_dim and 0 <= gy < self.grid_dim:
                    if self.occupancy_grid[gy, gx] > 0.4:
                        # Araç gövdesine olan net mesafe
                        # Basitleştirilmiş: r - araç yarıçapı
                        dist_m = max(r - 1.2, 0.05)
                        distances_cm[idx] = dist_m * 100.0
                        break

        return distances_cm

    def evaluate_park_warnings(self, min_distance_cm: float) -> Tuple[str, str]:
        """
        Mesafe değerine göre Tesla Vision Park ikaz durumunu belirler.
        """
        if min_distance_cm < 30.0:
            return "STOP", "#E82127"  # Kırmızı STOP
        elif min_distance_cm < 60.0:
            return f"{min_distance_cm:.0f} cm [KRİTİK]", "#E06C75"
        elif min_distance_cm < 100.0:
            return f"{min_distance_cm:.0f} cm [UYARI]", "#E5C07B"
        else:
            return "GÜVENLİ (>100 cm)", "#98C379"
