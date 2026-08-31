"""
Tesla Faz 4 Capstone Profilleyici Modülü
========================================
Bu modül; 8 Kamera, Radar, IMU, Odometri ve BEV Transformer birleşik hattının
uçtan uca yürütme gecikmesini ve takip hassasiyetini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_faz4_capstone_bev_fuzyon_hatti import TeslaPhase4CapstonePipeline


class TeslaCapstone4Profilleyici:
    """
    Faz 4 Capstone Füzyon Hattı Performans Profilleyicisi.
    """
    def __init__(self, steps: int = 100):
        self.steps = steps

    def benchmark_capstone_pipeline(self) -> Dict[str, Any]:
        pipeline = TeslaPhase4CapstonePipeline(bev_grid_size=60, bev_resolution_m=0.5)

        gecikmeler_us: List[float] = []
        lead_distances = []
        lead_speeds = []
        ego_x_list = []
        ego_y_list = []

        np.random.seed(42)
        for k in range(self.steps):
            # 8 Kamera Sentetik BEV İzdüşümleri
            camera_projections = {}
            for cam in pipeline.camera_names:
                grid = np.zeros((60, 60), dtype=np.float32)
                # Öndeki araç (30. hücre civarı)
                grid[28:32, 28:32] = 2.5 + np.random.normal(0, 0.2)
                # Yol şeritleri
                grid[:, 24] = 1.8
                grid[:, 36] = 1.8
                camera_projections[cam] = grid

            # Radar Ölçümü (25m mesafe, 0 rad, 15 m/s)
            z_radar = np.array([25.0 + np.random.normal(0, 0.1), 0.0, 15.0 + np.random.normal(0, 0.1)])

            # IMU ve Odometri
            imu_meas = (0.0, 0.005)
            wheel_meas = (15.0, 15.0)

            t0 = time.perf_counter_ns()
            res = pipeline.process_fsd_step(
                camera_projections_8cam=camera_projections,
                radar_measurement=z_radar,
                imu_measurement=imu_meas,
                wheel_speeds=wheel_meas,
                dt_s=0.0277
            )
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

            lead_distances.append(res["lead_distance_m"])
            lead_speeds.append(res["lead_speed_mps"])
            ego_x_list.append(res["dead_reckoning_pose"][0])
            ego_y_list.append(res["dead_reckoning_pose"][1])

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "pipeline_step_ortalama_us": t_avg_us,
            "pipeline_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_fsd_adimi": int(1e6 / max(t_avg_us, 1e-4)),
            "son_lead_mesafe_m": float(lead_distances[-1]),
            "son_lead_hiz_mps": float(lead_speeds[-1]),
            "son_ego_x_m": float(ego_x_list[-1]),
            "bev_occupancy": pipeline.bev_occupancy,
            "lead_distances": lead_distances,
            "lead_speeds": lead_speeds,
            "ego_x_list": ego_x_list,
            "gecikmeler": gecikmeler_us[:200]
        }
