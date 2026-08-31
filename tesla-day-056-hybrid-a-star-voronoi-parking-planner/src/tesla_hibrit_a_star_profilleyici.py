"""
Tesla Hibrit A* Profilleyici Modülü
===================================
Bu modül; Hibrit A* kinematik bisiklet adımı simülasyon hızını,
otonom park planlama gecikmesini ve park cebi konum hatasını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_hibrit_a_star_park_planlayici import TeslaHybridAStarParkPlanner


class TeslaHibritAStarProfilleyici:
    """
    Hibrit A* Park Planlayıcı Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_park_planner(self) -> Dict[str, Any]:
        planner = TeslaHybridAStarParkPlanner(wheelbase_m=2.875, max_steer_rad=0.55, dt_s=0.2)

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = planner.plan_parallel_parking_trajectory()
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "park_step_ortalama_us": t_avg_us,
            "park_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_park_plani": int(1e6 / max(t_avg_us, 1e-4)),
            "trajectory": ciktilar["trajectory"],
            "steering_cmds": ciktilar["steering_commands_rad"],
            "final_state": ciktilar["final_state"],
            "final_pos_err_m": ciktilar["final_pos_error_m"],
            "final_yaw_err_deg": ciktilar["final_yaw_error_deg"],
            "obstacles": ciktilar["obstacles"],
            "success": ciktilar["success"],
            "gecikmeler": gecikmeler_us[:200]
        }
