"""
Tesla Hız Profili Profilleyici Modülü
=====================================
Bu modül; İleri-Geri geçişli hız profili optimizasyon süresini,
viraj yanal ivme sınırlarını ve rejeneratif enerji geri kazanımını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_hiz_profili_optimize_edici import TeslaSpeedProfileOptimizer


class TeslaHizProfiliProfilleyici:
    """
    Hız Profili Optimizasyonu Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_speed_profiler(self) -> Dict[str, Any]:
        optimizer = TeslaSpeedProfileOptimizer()

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = optimizer.optimize_speed_profile()
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "speed_step_ortalama_us": t_avg_us,
            "speed_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_hiz_profili": int(1e6 / max(t_avg_us, 1e-4)),
            "s_array": ciktilar["s_array"],
            "v_opt": ciktilar["optimized_speed_mps"],
            "v_limits": ciktilar["speed_limits_mps"],
            "long_acc": ciktilar["longitudinal_acc_mps2"],
            "lat_acc": ciktilar["lateral_acc_mps2"],
            "regen_energy_kj": ciktilar["regen_energy_kj"],
            "min_corner_speed": ciktilar["min_corner_speed_mps"],
            "max_straight_speed": ciktilar["max_straight_speed_mps"],
            "is_comfortable": ciktilar["is_comfortable"],
            "gecikmeler": gecikmeler_us[:200]
        }
