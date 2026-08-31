"""
Tesla Optimus Profilleyici Modülü
=================================
Bu modül; Optimus 6-DoF ters dinamik ve empedans tork kontrolcüsünün
1000 Hz RTOS çevrim hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_optimus_eklem_kontrolcu import TeslaOptimusJointController


class TeslaOptimusProfilleyici:
    """
    Tesla Optimus Eklem Kontrol Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_optimus_joints(self) -> Dict[str, Any]:
        ctrl = TeslaOptimusJointController()

        q_curr = np.array([0.1, 0.2, -0.3, 0.4, -0.1, 0.05])
        q_dot = np.zeros(6)
        q_des = np.array([0.5, 0.6, 0.0, 0.8, 0.2, 0.1])

        gecikmeler_us: List[float] = []

        for _ in range(self.iterations):
            c_inst = TeslaOptimusJointController()
            t0 = time.perf_counter_ns()
            _ = c_inst.compute_inverse_dynamics_torque(q_curr, q_dot, np.full(6, 1.5))
            _ = c_inst.simulate_joint_step(q_curr, q_dot, q_des, dt_s=0.001)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        # 50 adımlık eklem yörünge takip simülasyonu
        q_sim = q_curr.copy()
        q_dot_sim = q_dot.copy()
        trajectory_error = []
        for _ in range(50):
            step_res = ctrl.simulate_joint_step(q_sim, q_dot_sim, q_des, dt_s=0.01)
            q_sim = np.array(step_res["q_next_rad"])
            q_dot_sim = np.array(step_res["q_dot_next_rad_s"])
            trajectory_error.append(step_res["pos_error_norm_rad"])

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        diag_res = ctrl.simulate_joint_step(q_curr, q_dot, q_des)

        return {
            "num_dof": ctrl.num_dof,
            "max_joint_torque_nm": diag_res["max_joint_torque_nm"],
            "initial_error_rad": float(np.linalg.norm(q_des - q_curr)),
            "final_error_rad": trajectory_error[-1],
            "step_ortalama_us": t_avg_us,
            "step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_rtos_frekansi": int(1e6 / max(t_avg_us, 1e-4)),
            "trajectory_error": trajectory_error,
            "torques_sample": diag_res["tau_cmd_nm"],
            "gecikmeler": gecikmeler_us[:200]
        }
