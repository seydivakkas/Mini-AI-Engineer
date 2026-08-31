"""
Tesla 3D Occupancy Profilleyici Modülü
======================================
Bu modül; 40,000 hücreli 3D Voksel ızgarasının doluluk çıkarım hızını,
3D Voxel Flow hız alanı sorgulama gecikmesini ve bellek tüketimini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_3d_occupancy_network import Tesla3DOccupancyNetwork


class TeslaOccupancyProfilleyici:
    """
    3D Voksel Doluluk Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_occupancy_network(self) -> Dict[str, Any]:
        occ_net = Tesla3DOccupancyNetwork(grid_dim_x=50, grid_dim_y=50, grid_dim_z=16)
        occ_net.insert_synthetic_scene()

        gecikmeler_us: List[float] = []

        probs, binary_mask = None, None
        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            probs, binary_mask = occ_net.compute_occupancy_probabilities(threshold=0.5)
            # Rastgele 10 nokta hız sorgusu
            for _ in range(10):
                occ_net.query_point_velocity(x_m=15.0, y_m=0.0, z_m=1.0)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        # Araç ve Yaya Hız Sorguları
        car_prob, car_flow = occ_net.query_point_velocity(x_m=15.0, y_m=0.0, z_m=1.0)
        ped_prob, ped_flow = occ_net.query_point_velocity(x_m=5.0, y_m=6.0, z_m=1.0)
        tree_prob, _ = occ_net.query_point_velocity(x_m=20.0, y_m=0.0, z_m=1.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        total_voxels = occ_net.nx * occ_net.ny * occ_net.nz
        occupied_count = int(np.sum(binary_mask))

        return {
            "occupancy_step_ortalama_us": t_avg_us,
            "occupancy_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_voksel_adimi": int(1e6 / max(t_avg_us, 1e-4)),
            "toplam_voksel": total_voxels,
            "dolu_voksel_sayisi": occupied_count,
            "doluluk_orani_pct": float(occupied_count / total_voxels * 100.0),
            "car_vx_mps": float(car_flow[0]),
            "ped_vy_mps": float(ped_flow[1]),
            "tree_captured": bool(tree_prob > 0.5),
            "bev_projection": np.max(probs, axis=2),
            "gecikmeler": gecikmeler_us[:200]
        }
