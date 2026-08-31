"""
Tesla Clothoid Profilleyici Modülü
==================================
Bu modül; Clothoid yörünge üretim hızını, eğrilik sürekliliği hesaplama süresini
ve dinamik engelden kaçınma gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_clothoid_kacinma_planlayici import TeslaClothoidAvoidancePlanner


class TeslaClothoidProfilleyici:
    """
    Clothoid Yörünge Planlayıcı Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_clothoid_planner(self) -> Dict[str, Any]:
        planner = TeslaClothoidAvoidancePlanner()

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = planner.plan_obstacle_avoidance_maneuver(obstacle_x_m=35.0, obstacle_y_m=0.0)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "clothoid_step_ortalama_us": t_avg_us,
            "clothoid_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_clothoid_plani": int(1e6 / max(t_avg_us, 1e-4)),
            "x_traj": ciktilar["x_traj"],
            "y_traj": ciktilar["y_traj"],
            "curvature_kappa": ciktilar["curvature_kappa"],
            "theta_traj": ciktilar["theta_traj"],
            "min_clearance_m": ciktilar["min_clearance_m"],
            "obstacle_pos": ciktilar["obstacle_pos"],
            "is_safe": ciktilar["is_safe"],
            "gecikmeler": gecikmeler_us[:200]
        }
