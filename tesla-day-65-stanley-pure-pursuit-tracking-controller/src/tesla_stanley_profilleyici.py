"""
Tesla Stanley Profilleyici Modülü
=================================
Bu modül; Stanley ve Pure Pursuit kontrol hesaplama süresini,
kapalı çevrim yol takip hatası yakınsamasını ve RTOS gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_stanley_pure_pursuit_kontrolcu import TeslaStanleyTracker, TeslaTrackingBenchmark


class TeslaStanleyProfilleyici:
    """
    Stanley Takip Kontrolcüsü Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_tracker(self) -> Dict[str, Any]:
        sim = TeslaTrackingBenchmark()

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = sim.run_tracking_simulation(steps=50, speed_mps=15.0)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "stanley_step_ortalama_us": t_avg_us,
            "stanley_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_takip_cevrimi": int(1e6 / max(t_avg_us, 1e-4)),
            "errors": ciktilar["stanley_errors_m"],
            "steers": ciktilar["stanley_steers_rad"],
            "final_err": ciktilar["final_lateral_error_m"],
            "is_converged": ciktilar["is_converged"],
            "gecikmeler": gecikmeler_us[:200]
        }
