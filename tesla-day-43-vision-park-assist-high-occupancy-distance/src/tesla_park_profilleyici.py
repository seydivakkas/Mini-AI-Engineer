"""
Tesla Vision Park Asistanı Profilleyici Modülü
==============================================
Bu modül; 3D Voxel doluluk güncellemesini, 360° ışın atma (Ray-Casting) mesafe
kestirimini ve kör nokta hafıza korumasını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_vision_park_asistani import TeslaVisionParkAssist


class TeslaParkProfilleyici:
    """
    Tesla Vision Park Asistanı Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 50):
        self.iterations = iterations

    def benchmark_park_assist(self) -> Dict[str, Any]:
        park = TeslaVisionParkAssist(grid_resolution_m=0.05, grid_size_m=10.0)

        # Sentetik Park Senaryosu:
        # Sağda Kaldırım (Y = -1.5m), Arkada Duvar (X = -2.8m), Önde Araç (X = +3.0m)
        points_curb = np.column_stack([np.linspace(-3, 3, 50), np.full(50, -1.6)])
        points_wall = np.column_stack([np.full(40, -2.6), np.linspace(-1.5, 1.5, 40)])
        points_front_car = np.column_stack([np.full(40, 2.9), np.linspace(-1.0, 1.0, 40)])

        all_points = np.vstack([points_curb, points_wall, points_front_car])

        gecikmeler_us: List[float] = []

        distances_360 = None
        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            # 5 cm geri yanaşma hareketi ve nokta bulutu güncellemesi
            park.update_occupancy_and_memory(all_points, ego_delta_x=-0.05, ego_delta_y=0.0)
            distances_360 = park.compute_360_distance_contour(num_angles=360)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        # En yakın mesafe ve ikaz
        valid_distances = distances_360[distances_360 < 900.0]
        min_dist_cm = float(np.min(valid_distances)) if len(valid_distances) > 0 else 150.0
        warning_text, warning_color = park.evaluate_park_warnings(min_dist_cm)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "park_step_ortalama_us": t_avg_us,
            "park_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_park_guncellemesi": int(1e6 / max(t_avg_us, 1e-4)),
            "min_mesafe_cm": min_dist_cm,
            "ikaz_metni": warning_text,
            "ikaz_rengi": warning_color,
            "occupancy_grid": park.occupancy_grid,
            "distances_360": distances_360,
            "gecikmeler": gecikmeler_us[:200]
        }
