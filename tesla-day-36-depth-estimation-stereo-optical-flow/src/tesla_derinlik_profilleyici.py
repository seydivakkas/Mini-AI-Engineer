"""
Tesla Derinlik ve Optik Akış Profilleyici Modülü
================================================
Bu modül; Disparity-to-Depth dönüşümünü, derinlik belirsizliğini,
Lucas-Kanade optik akış çözümleme hızını ve TTC güvenliğini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_derinlik_ve_optik_akis import TeslaDepthAndOpticalFlowEstimator


class TeslaDerinlikProfilleyici:
    """
    Derinlik ve Optik Akış Performans Profilleyicisi.
    """
    def __init__(self, num_samples: int = 500, iterations: int = 100):
        self.num_samples = num_samples
        self.iterations = iterations

    def benchmark_derinlik_ve_akis(self) -> Dict[str, Any]:
        estimator = TeslaDepthAndOpticalFlowEstimator(focal_length_px=1200.0, baseline_m=0.50)

        # 5 metreden 100 metreye kadar sentetik derinlikler
        z_true = np.linspace(5.0, 100.0, self.num_samples)
        # Disparity = (f * B) / Z
        disp_true = (estimator.f_px * estimator.b_m) / z_true
        # 0.2 px Gauss gürültüsü ekle
        np.random.seed(42)
        disp_noisy = disp_true + np.random.normal(0, 0.2, self.num_samples)

        gecikmeler_us: List[float] = []

        z_estimated = None
        uncertainty = None
        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            z_estimated = estimator.disparity_to_depth(disp_noisy)
            uncertainty = estimator.compute_depth_uncertainty(z_estimated)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        # Lucas-Kanade Test Yaması
        patch1 = np.random.uniform(0, 255, (15, 15))
        patch2 = np.roll(patch1, shift=(1, 2), axis=(0, 1))  # vx=2, vy=1
        vx, vy = estimator.estimate_lucas_kanade_flow(patch1, patch2)

        # TTC Testi: 30 metrede 15 m/s yaklaşan araç
        ttc_sec = estimator.compute_time_to_contact(depth_m=30.0, rel_speed_mps=15.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))
        mae_depth = float(np.mean(np.abs(z_estimated - z_true)))

        return {
            "derinlik_step_ortalama_us": t_avg_us,
            "derinlik_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_derinlik_haritasi": int(1e6 / max(t_avg_us, 1e-4)),
            "mae_depth_m": mae_depth,
            "ttc_sec": ttc_sec,
            "lk_vx": vx,
            "lk_vy": vy,
            "z_true": z_true,
            "z_estimated": z_estimated,
            "uncertainty": uncertainty,
            "disparities": disp_noisy,
            "gecikmeler": gecikmeler_us[:200]
        }
