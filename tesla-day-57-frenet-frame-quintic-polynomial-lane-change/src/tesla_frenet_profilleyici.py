"""
Tesla Frenet Profilleyici Modülü
================================
Bu modül; Quintic Polinom çözme hızını, Jerk/İvme türev hesaplama süresini
ve Frenet şerit değiştirme planlama gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_frenet_ve_quintic_serit_degistirme import TeslaFrenetTrajectoryPlanner


class TeslaFrenetProfilleyici:
    """
    Frenet ve Quintic Polinom Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_frenet_planner(self) -> Dict[str, Any]:
        planner = TeslaFrenetTrajectoryPlanner(target_lane_width_m=3.5, time_horizon_s=4.0)

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = planner.generate_frenet_lane_change(current_speed_mps=25.0, steps=50)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "frenet_step_ortalama_us": t_avg_us,
            "frenet_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_frenet_plani": int(1e6 / max(t_avg_us, 1e-4)),
            "time_arr": ciktilar["time_array"],
            "long_s": ciktilar["longitudinal_s"],
            "profiles": ciktilar["profiles"],
            "coeffs": ciktilar["coeffs"],
            "max_jerk": ciktilar["max_lateral_jerk"],
            "max_acc": ciktilar["max_lateral_acc"],
            "is_comfortable": ciktilar["is_comfortable"],
            "gecikmeler": gecikmeler_us[:200]
        }
