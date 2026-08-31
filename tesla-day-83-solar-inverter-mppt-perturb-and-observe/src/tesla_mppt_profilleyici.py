"""
Tesla Solar MPPT Profilleyici Modülü
====================================
Bu modül; Perturb and Observe MPPT algoritmasının adım hesaplama hızını ve
Maksimum Güç Noktası Takip gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_solar_mppt_kontrolcu import TeslaSolarMPPTController


class TeslaSolarMPPTProfilleyici:
    """
    Tesla Solar Inverter MPPT Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_mppt(self) -> Dict[str, Any]:
        ctrl = TeslaSolarMPPTController()

        gecikmeler_us: List[float] = []

        for _ in range(self.iterations):
            c_inst = TeslaSolarMPPTController()
            t0 = time.perf_counter_ns()
            _ = c_inst.calculate_pv_power(38.5)
            _ = c_inst.mppt_step_perturb_and_observe(38.5, 355.0, step_v=0.5)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        sim_ctrl = TeslaSolarMPPTController()
        sim_res = sim_ctrl.simulate_mppt_tracking(initial_v=15.0, iterations=60, step_v=0.5)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "step_ortalama_us": t_avg_us,
            "step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_mppt_frekansi": int(1e6 / max(t_avg_us, 1e-4)),
            "optimal_p": sim_res["optimal_p_mpp"],
            "tracked_p": sim_res["final_tracked_p"],
            "efficiency": sim_res["mppt_efficiency_pct"],
            "v_hist": sim_res["v_history"],
            "p_hist": sim_res["p_history"],
            "locked": sim_res["locked_on_mpp"],
            "gecikmeler": gecikmeler_us[:200]
        }
