"""
Tesla MPC Profilleyici Modülü
==============================
Bu modül; Model Predictive Control Riccati denklemi çözme hızını,
kapalı çevrim simülasyon gecikmesini ve hata yakınsama performansını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_mpc_kinematik_kontrolcu import TeslaKinematicMPCController


class TeslaMPCProfilleyici:
    """
    MPC Kinematik Kontrolcü Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_mpc_controller(self) -> Dict[str, Any]:
        controller = TeslaKinematicMPCController()

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = controller.simulate_closed_loop_tracking()
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "mpc_step_ortalama_us": t_avg_us,
            "mpc_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_mpc_cevrimi": int(1e6 / max(t_avg_us, 1e-4)),
            "lat_errors": ciktilar["lateral_errors_m"],
            "yaw_errors": ciktilar["heading_errors_rad"],
            "steer_cmds": ciktilar["steer_cmds_rad"],
            "acc_cmds": ciktilar["acc_cmds_mps2"],
            "final_lat_err": ciktilar["final_lat_err_m"],
            "final_yaw_err_deg": ciktilar["final_yaw_err_deg"],
            "is_converged": ciktilar["is_converged"],
            "gecikmeler": gecikmeler_us[:200]
        }
