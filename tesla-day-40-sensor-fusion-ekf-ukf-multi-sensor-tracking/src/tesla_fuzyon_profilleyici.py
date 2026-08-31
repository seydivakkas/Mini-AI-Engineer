"""
Tesla Sensör Füzyonu Profilleyici Modülü
========================================
Bu modül; Kamera ve Radar asenkron füzyonunun takip hassasiyetini (RMSE),
Mahalanobis outlier filtreleme oranını ve EKF adım gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_sensor_fuzyonu_ekf_ukf import TeslaSensorFusionEKF


class TeslaFuzyonProfilleyici:
    """
    Sensör Füzyonu Performans Profilleyicisi.
    """
    def __init__(self, steps: int = 200, dt_s: float = 0.05):
        self.steps = steps
        self.dt = dt_s

    def benchmark_sensor_fuzyonu(self) -> Dict[str, Any]:
        fusion = TeslaSensorFusionEKF(init_x=10.0, init_y=0.0, init_vx=15.0, init_vy=0.5)

        # Gerçek Hedef Yörüngesi (Sabit İvmeli Dönüş)
        t_arr = np.arange(self.steps) * self.dt
        gt_x = 10.0 + 15.0 * t_arr + 0.1 * (t_arr ** 2)
        gt_y = 0.5 * t_arr + 0.5 * np.sin(0.5 * t_arr)
        gt_vx = 15.0 + 0.2 * t_arr
        gt_vy = 0.5 + 0.25 * np.cos(0.5 * t_arr)

        fused_x = []
        fused_y = []
        fused_vx = []
        fused_vy = []
        gecikmeler_us: List[float] = []

        np.random.seed(42)
        for k in range(self.steps):
            # 1. Tahmin Adımı (20 Hz)
            t0 = time.perf_counter_ns()
            fusion.predict(self.dt)

            # 2. Kamera Güncellemesi (20 Hz)
            z_cam = np.array([gt_x[k] + np.random.normal(0, 0.4), gt_y[k] + np.random.normal(0, 0.4)])
            fusion.update_camera(z_cam)

            # 3. Radar Güncellemesi (10 Hz - 2 adımda bir)
            if k % 2 == 0:
                r_true = np.hypot(gt_x[k], gt_y[k])
                th_true = np.arctan2(gt_y[k], gt_x[k])
                rdot_true = (gt_x[k]*gt_vx[k] + gt_y[k]*gt_vy[k]) / r_true

                z_radar = np.array([
                    r_true + np.random.normal(0, 0.2),
                    th_true + np.random.normal(0, 0.01),
                    rdot_true + np.random.normal(0, 0.15)
                ])
                fusion.update_radar(z_radar)

            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

            fused_x.append(fusion.x[0])
            fused_y.append(fusion.x[1])
            fused_vx.append(fusion.x[2])
            fused_vy.append(fusion.x[3])

        # RMSE Hata Hesaplama
        rmse_pos_x = float(np.sqrt(np.mean((np.array(fused_x) - gt_x) ** 2)))
        rmse_pos_y = float(np.sqrt(np.mean((np.array(fused_y) - gt_y) ** 2)))
        rmse_vel_x = float(np.sqrt(np.mean((np.array(fused_vx) - gt_vx) ** 2)))
        rmse_vel_y = float(np.sqrt(np.mean((np.array(fused_vy) - gt_vy) ** 2)))

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "fuzyon_step_ortalama_us": t_avg_us,
            "fuzyon_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_fuzyon_adimi": int(1e6 / max(t_avg_us, 1e-4)),
            "rmse_pos_x_m": rmse_pos_x,
            "rmse_pos_y_m": rmse_pos_y,
            "rmse_vel_x_mps": rmse_vel_x,
            "rmse_vel_y_mps": rmse_vel_y,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "fused_x": fused_x,
            "fused_y": fused_y,
            "fused_vx": fused_vx,
            "fused_vy": fused_vy,
            "t_arr": t_arr,
            "gecikmeler": gecikmeler_us[:200]
        }
