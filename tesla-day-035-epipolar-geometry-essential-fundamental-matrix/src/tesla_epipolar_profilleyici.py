"""
Tesla Epipolar Geometri Profilleyici Modülü
===========================================
Bu modül; 2 kamera arasındaki Essential ve Fundamental matris çözümleme hızını,
8-nokta SVD algoritmasını ve Sampson geometrik hata profilini ölçer.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_epipolar_geometri_ve_matris import TeslaEpipolarCalibrator


class TeslaEpipolarProfilleyici:
    """
    Epipolar Geometri ve Çoklu Görüş Profilleyicisi.
    """
    def __init__(self, num_points: int = 50, iterations: int = 100):
        self.num_points = num_points
        self.iterations = iterations

    def benchmark_epipolar_kalibrasyon(self) -> Dict[str, Any]:
        # İki Kamera İçsel Parametreleri (1280x960)
        K1 = np.array([[1200.0, 0, 640.0], [0, 1200.0, 480.0], [0, 0, 1.0]])
        K2 = np.array([[1200.0, 0, 640.0], [0, 1200.0, 480.0], [0, 0, 1.0]])

        # Stereo Taban Çizgisi: 50 cm sağa öteleme (X ekseni) ve 2 derece yaw rotasyonu
        yaw_rad = np.radians(2.0)
        R = np.array([[np.cos(yaw_rad), 0, np.sin(yaw_rad)], [0, 1, 0], [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]])
        t = np.array([0.50, 0.0, 0.0])  # 50 cm stereo baseline

        # Analitik F ve E matrisleri
        E_true = TeslaEpipolarCalibrator.compute_essential_matrix(R, t)
        F_true = TeslaEpipolarCalibrator.compute_fundamental_matrix(K1, K2, R, t)

        # 3D Noktalar Üret ve İki Kameraya İzdüşür
        np.random.seed(42)
        pts_3d = np.random.uniform(low=[-5, -2, 10], high=[5, 2, 30], size=(self.num_points, 3))

        pts_cam1 = []
        pts_cam2 = []
        for p in pts_3d:
            # Cam 1: P
            p1_cam = p
            u1 = K1[0, 0] * (p1_cam[0] / p1_cam[2]) + K1[0, 2]
            v1 = K1[1, 1] * (p1_cam[1] / p1_cam[2]) + K1[1, 2]
            pts_cam1.append([u1, v1])

            # Cam 2: R @ (P - t)
            p2_cam = R @ (p - t)
            u2 = K2[0, 0] * (p2_cam[0] / p2_cam[2]) + K2[0, 2]
            v2 = K2[1, 1] * (p2_cam[1] / p2_cam[2]) + K2[1, 2]
            pts_cam2.append([u2, v2])

        pts_cam1_arr = np.array(pts_cam1)
        pts_cam2_arr = np.array(pts_cam2)

        # 8-Nokta ile F Kestirimi ve Profilleme
        gecikmeler_us: List[float] = []
        F_est = None
        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            F_est = TeslaEpipolarCalibrator.estimate_fundamental_8point(pts_cam1_arr[:8], pts_cam2_arr[:8])
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        # Sampson Geometrik Hata Hesaplama
        sampson_errors = [
            TeslaEpipolarCalibrator.compute_sampson_distance(F_true, p1, p2)
            for p1, p2 in zip(pts_cam1_arr, pts_cam2_arr)
        ]

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "epipolar_step_ortalama_us": t_avg_us,
            "epipolar_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_cozum_sayisi": int(1e6 / max(t_avg_us, 1e-4)),
            "sampson_error_mean_px": float(np.mean(sampson_errors)),
            "sampson_error_max_px": float(np.max(sampson_errors)),
            "rank_F": int(np.linalg.matrix_rank(F_est)),
            "det_F": float(np.linalg.det(F_est)),
            "E_matrix": E_true,
            "F_matrix": F_true,
            "sampson_errors": sampson_errors,
            "pts_cam1": pts_cam1_arr,
            "pts_cam2": pts_cam2_arr,
            "gecikmeler": gecikmeler_us[:200]
        }
