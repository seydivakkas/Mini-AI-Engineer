"""
Tesla ZMP Profilleyici Modülü
=============================
Bu modül; Optimus ZMP ve Capture Point denge algoritmalarının RTOS
çözümleme hızını ve gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_optimus_zmp_denge_kontrolcu import TeslaOptimusZMPBalanceController


class TeslaZMPProfilleyici:
    """
    Tesla Optimus ZMP Denge Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_zmp_balance(self) -> Dict[str, Any]:
        ctrl = TeslaOptimusZMPBalanceController()

        gecikmeler_us: List[float] = []

        for _ in range(self.iterations):
            c_inst = TeslaOptimusZMPBalanceController()
            t0 = time.perf_counter_ns()
            _ = c_inst.compute_zmp(0.02, -0.01, 0.4, -0.2)
            _ = c_inst.compute_capture_point(0.02, -0.01, 0.15, -0.05)
            _ = c_inst.push_recovery_step(0.0, 0.0, 0.0, 0.0, ext_impulse_ns=45.0)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        # 50 adımlık yürüyüş ZMP ve CoM yörüngesi simülasyonu
        t_vec = np.linspace(0, 2.0, 50)
        x_com_traj = 0.05 * np.sin(2.0 * np.pi * t_vec)
        y_com_traj = 0.03 * np.cos(2.0 * np.pi * t_vec)
        x_zmp_traj = []
        y_zmp_traj = []

        for i in range(len(t_vec)):
            x_ddot = -0.05 * (2.0 * np.pi)**2 * np.sin(2.0 * np.pi * t_vec[i])
            y_ddot = -0.03 * (2.0 * np.pi)**2 * np.cos(2.0 * np.pi * t_vec[i])
            xz, yz = ctrl.compute_zmp(x_com_traj[i], y_com_traj[i], x_ddot, y_ddot)
            x_zmp_traj.append(xz)
            y_zmp_traj.append(yz)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))
        push_res = ctrl.push_recovery_step(0.0, 0.0, 0.0, 0.0, ext_impulse_ns=50.0)

        return {
            "robot_mass_kg": ctrl.mass,
            "com_height_m": ctrl.z_com,
            "natural_freq_rad_s": round(ctrl.omega_0, 3),
            "step_ortalama_us": t_avg_us,
            "step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_denge_frekansi": int(1e6 / max(t_avg_us, 1e-4)),
            "x_com_traj": list(x_com_traj),
            "y_com_traj": list(y_com_traj),
            "x_zmp_traj": x_zmp_traj,
            "y_zmp_traj": y_zmp_traj,
            "push_res": push_res,
            "gecikmeler": gecikmeler_us[:200]
        }
