"""
Tesla Yörünge Profilleyici Modülü
=================================
Bu modül; Çoklu Modal Gelecek Yörünge Tahmini ve TTC hesaplama gecikmelerini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_yorunge_tahmin_lstm_difuzyon import TeslaTrajectoryPredictor


class TeslaYorungeProfilleyici:
    """
    Yörünge Tahmini Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_trajectory_predictor(self) -> Dict[str, Any]:
        predictor = TeslaTrajectoryPredictor(horizon_steps=50, dt_s=0.1)

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = predictor.predict_multi_modal_trajectories()
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "yorunge_step_ortalama_us": t_avg_us,
            "yorunge_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_tahmin_adimi": int(1e6 / max(t_avg_us, 1e-4)),
            "trajectories": ciktilar["trajectories"],
            "probabilities": ciktilar["probabilities"],
            "modes": ciktilar["modes"],
            "ttc_sec": ciktilar["ttc_seconds"],
            "min_dist": ciktilar["min_distance_m"],
            "gecikmeler": gecikmeler_us[:200]
        }
