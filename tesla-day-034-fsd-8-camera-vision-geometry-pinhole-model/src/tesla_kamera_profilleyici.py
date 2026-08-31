"""
Tesla 8-Kamera Görüş Geometrisi Profilleyici Modülü
===================================================
Bu modül; 8 kameranın 3D sahne izdüşüm hızını, Brown-Conrady distorsiyon
düzeltme maliyetini ve 36 FPS FSD gerçek zamanlı bütçesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_8kamera_gorus_geometrisi import Tesla8CameraVisionRig


class TeslaKameraProfilleyici:
    """
    Tesla FSD 8-Kamera Geometri Profilleyicisi.
    """
    def __init__(self, num_points: int = 200, iterations: int = 100):
        self.num_points = num_points
        self.iterations = iterations

    def benchmark_kamera_geometrisi(self) -> Dict[str, Any]:
        rig = Tesla8CameraVisionRig()

        # 360 derece etrafta rastgele 3D noktalar üret
        np.random.seed(42)
        angles = np.random.uniform(0, 2*np.pi, self.num_points)
        radii = np.random.uniform(5.0, 60.0, self.num_points)
        z_vals = np.random.uniform(-0.5, 3.0, self.num_points)

        points_ego = [
            np.array([r * np.cos(a), r * np.sin(a), z])
            for a, r, z in zip(angles, radii, z_vals)
        ]

        gecikmeler_frame_us: List[float] = []

        last_projection = {}
        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            last_projection = rig.project_3d_scene(points_ego)
            t1 = time.perf_counter_ns()
            gecikmeler_frame_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_frame_us)
        t_avg_us = float(np.mean(dizi))

        # Kamera bazlı görünür nokta sayıları
        cam_visibility_counts = {name: len(dets) for name, dets in last_projection.items()}
        total_visible_detections = sum(cam_visibility_counts.values())

        return {
            "geometri_step_ortalama_us": t_avg_us,
            "geometri_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_kare_isleme": int(1e6 / max(t_avg_us, 1e-4)),
            "total_points": self.num_points,
            "total_visible_detections": total_visible_detections,
            "cam_visibility_counts": cam_visibility_counts,
            "last_projection": last_projection,
            "gecikmeler": gecikmeler_frame_us[:200]
        }
