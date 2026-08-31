"""
Tesla IMU ve Odometri Profilleyici Modülü
=========================================
Bu modül; 100 Hz IMU + Tekerlek Füzyonunun konum doğruluğunu,
Jiroskop bias kestirimini ve saf IMU sürüklenmesine (Drift) karşı iyileşmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_imu_ve_odometri_fuzyonu import TeslaIMUWheelOdometryFusion


class TeslaIMUProfilleyici:
    """
    IMU ve Odometri Performans Profilleyicisi.
    """
    def __init__(self, steps: int = 500, dt_s: float = 0.01):
        self.steps = steps
        self.dt = dt_s

    def benchmark_dead_reckoning(self) -> Dict[str, Any]:
        fusion = TeslaIMUWheelOdometryFusion()

        # Dairesel Viraj Manevrası (v = 20 m/s = 72 km/h, yaw_rate = 0.1 rad/s)
        t_arr = np.arange(self.steps) * self.dt
        v_true = 20.0
        yaw_rate_true = 0.10
        gyro_bias_true = 0.008  # 0.008 rad/s donanımsal jiroskop kayması

        gt_psi = yaw_rate_true * t_arr
        gt_x = (v_true / yaw_rate_true) * np.sin(gt_psi)
        gt_y = (v_true / yaw_rate_true) * (1.0 - np.cos(gt_psi))

        fused_x = []
        fused_y = []
        fused_psi = []
        estimated_bias = []
        pure_imu_x = []
        pure_imu_y = []

        # Saf IMU durumu
        p_x, p_y, p_psi, p_v = 0.0, 0.0, 0.0, 20.0

        gecikmeler_us: List[float] = []

        np.random.seed(42)
        for k in range(self.steps):
            # Sensör Girdileri
            ax_meas = 0.0 + np.random.normal(0, 0.05)
            gyro_meas = yaw_rate_true + gyro_bias_true + np.random.normal(0, 0.002)

            # Tekerlek Hızları: v_R = v + (W/2)*yaw, v_L = v - (W/2)*yaw
            w_diff = (fusion.w_track / 2.0) * yaw_rate_true
            v_r_meas = (v_true + w_diff) + np.random.normal(0, 0.1)
            v_l_meas = (v_true - w_diff) + np.random.normal(0, 0.1)

            t0 = time.perf_counter_ns()
            fusion.predict_imu(ax_meas, gyro_meas, self.dt)
            fusion.update_wheel_odometry(v_l_meas, v_r_meas)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

            # Saf IMU Entegrasyonu (Bias düzeltmesiz - Sürüklenir)
            p_psi += gyro_meas * self.dt
            p_x += p_v * np.cos(p_psi) * self.dt
            p_y += p_v * np.sin(p_psi) * self.dt
            pure_imu_x.append(p_x)
            pure_imu_y.append(p_y)

            fused_x.append(fusion.x[0])
            fused_y.append(fusion.x[1])
            fused_psi.append(fusion.x[2])
            estimated_bias.append(fusion.x[4])

        # Hata Değerlendirmesi
        fused_err_m = np.hypot(np.array(fused_x) - gt_x, np.array(fused_y) - gt_y)
        pure_err_m = np.hypot(np.array(pure_imu_x) - gt_x, np.array(pure_imu_y) - gt_y)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "imu_step_ortalama_us": t_avg_us,
            "imu_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_dead_reckoning": int(1e6 / max(t_avg_us, 1e-4)),
            "final_fused_error_m": float(fused_err_m[-1]),
            "final_pure_error_m": float(pure_err_m[-1]),
            "drift_reduction_pct": float((1.0 - (fused_err_m[-1] / pure_err_m[-1])) * 100.0),
            "estimated_bias_final": float(estimated_bias[-1]),
            "gt_x": gt_x,
            "gt_y": gt_y,
            "fused_x": fused_x,
            "fused_y": fused_y,
            "pure_imu_x": pure_imu_x,
            "pure_imu_y": pure_imu_y,
            "t_arr": t_arr,
            "fused_err_m": fused_err_m,
            "pure_err_m": pure_err_m,
            "gecikmeler": gecikmeler_us[:200]
        }
