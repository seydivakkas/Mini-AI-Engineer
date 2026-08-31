"""
Tesla AEB Profilleyici Modülü
=============================
Bu modül; Otomatik Acil Frenleme (AEB) karar tetikleme hızını,
durma mesafesi hesaplama süresini ve AES kaçınma gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_aeb_ve_kacinma_manevrasi import TeslaAEBController


class TeslaAEBProfilleyici:
    """
    AEB ve Acil Durum Kontrolcü Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_aeb_controller(self) -> Dict[str, Any]:
        controller = TeslaAEBController()

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = controller.evaluate_aeb_trigger(
                ego_speed_mps=20.0,
                dist_to_obstacle_m=18.0,
                rel_speed_mps=20.0,
                is_adjacent_lane_clear=False
            )
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "aeb_step_ortalama_us": t_avg_us,
            "aeb_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_aeb_cevrimi": int(1e6 / max(t_avg_us, 1e-4)),
            "level": ciktilar["aeb_level"],
            "ttc_s": ciktilar["ttc_s"],
            "stopping_dist_m": ciktilar["stopping_dist_m"],
            "dist_obs_m": ciktilar["dist_to_obstacle_m"],
            "target_acc": ciktilar["target_acc_mps2"],
            "action_desc": ciktilar["action_desc"],
            "is_emergency": ciktilar["is_emergency"],
            "gecikmeler": gecikmeler_us[:200]
        }
