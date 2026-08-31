"""
Tesla FOC Motor Kontrolcü Profilleyici Modülü
==============================================
Bu modül; 0'dan 350 Nm tam tork ivmelenme basamağında FOC akım kontrol
döngüsünü (Clarke/Park/Ters Park ve PI) ve 10 kHz RTOS döngü hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_foc_motor_kontrolcusu import (
    TeslaMotorParameters,
    TeslaFOCController
)


class TeslaFOCProfilleyici:
    """
    FOC ve PMSM Motor Performans Profilleyicisi.
    """
    def __init__(self, sim_adimlari: int = 1000):
        self.sim_adimlari = sim_adimlari

    def benchmark_foc_dongusu(self) -> Dict[str, Any]:
        motor_params = TeslaMotorParameters()
        foc = TeslaFOCController(motor_params)

        target_torque_history = []
        actual_torque_history = []
        i_a_history = []
        i_b_history = []
        i_c_history = []
        i_d_history = []
        i_q_history = []
        v_a_history = []
        gecikmeler_foc_us: List[float] = []

        # Simülasyon: 10 kHz (dt = 100 µs), Rotor Hızı = 3000 RPM
        rpm = 3000.0
        omega_m = rpm * (2.0 * np.pi / 60.0)
        omega_e = omega_m * motor_params.pole_pairs  # Elektriksel açısal hız
        dt = 0.0001
        theta_e = 0.0

        current_iq_sim = 0.0
        current_id_sim = 0.0

        for step in range(self.sim_adimlari):
            theta_e += omega_e * dt
            theta_e = float(theta_e % (2.0 * np.pi))

            # 200. adımda 350 Nm tam tork basamağı
            target_t = 350.0 if step >= 200 else 50.0

            # Basit motor akım dinamiği simülasyonu
            kt = 1.5 * motor_params.pole_pairs * motor_params.psi_f_wb
            target_iq = target_t / kt
            current_iq_sim += (target_iq - current_iq_sim) * 0.15
            current_id_sim += (0.0 - current_id_sim) * 0.15

            # dq'dan 3 faza dönüşüm (Geri besleme sensör akımı)
            i_alpha_sim = current_id_sim * np.cos(theta_e) - current_iq_sim * np.sin(theta_e)
            i_beta_sim = current_id_sim * np.sin(theta_e) + current_iq_sim * np.cos(theta_e)
            i_a_sim = i_alpha_sim
            i_b_sim = -0.5 * i_alpha_sim + (np.sqrt(3.0) / 2.0) * i_beta_sim
            i_c_sim = -0.5 * i_alpha_sim - (np.sqrt(3.0) / 2.0) * i_beta_sim

            t0 = time.perf_counter_ns()
            out = foc.execute_foc_step(
                target_torque_nm=target_t,
                i_a=i_a_sim,
                i_b=i_b_sim,
                i_c=i_c_sim,
                rotor_theta_e_rad=theta_e,
                dt_s=dt
            )
            t1 = time.perf_counter_ns()
            gecikmeler_foc_us.append(float(t1 - t0) / 1000.0)

            target_torque_history.append(target_t)
            actual_torque_history.append(out["actual_torque_nm"])
            i_a_history.append(i_a_sim)
            i_b_history.append(i_b_sim)
            i_c_history.append(i_c_sim)
            i_d_history.append(out["i_d"])
            i_q_history.append(out["i_q"])
            v_a_history.append(out["v_a"])

        foc_dizi = np.array(gecikmeler_foc_us)
        t_foc_avg_us = float(np.mean(foc_dizi))

        return {
            "foc_step_ortalama_us": t_foc_avg_us,
            "foc_step_p99_us": float(np.percentile(foc_dizi, 99)),
            "saniyelik_foc_adimi": int(1e6 / max(t_foc_avg_us, 1e-4)),
            "max_torque_nm": float(np.max(actual_torque_history)),
            "target_torque": target_torque_history,
            "actual_torque": actual_torque_history,
            "i_a": i_a_history,
            "i_b": i_b_history,
            "i_c": i_c_history,
            "i_d": i_d_history,
            "i_q": i_q_history,
            "v_a": v_a_history,
            "foc_gecikmeler": gecikmeler_foc_us[:200]
        }
