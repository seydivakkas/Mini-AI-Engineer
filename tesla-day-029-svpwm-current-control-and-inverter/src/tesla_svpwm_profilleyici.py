"""
Tesla SVPWM Modülatör Profilleyici Modülü
=========================================
Bu modül; 1 tam elektriksel periyot boyunca 6 sektörün SVPWM görev çevrimlerini,
SVPWM'in SPWM'e göre %15.47 DC bara voltaj kazancını ve modülasyon hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_svpwm_modulatoru import TeslaSVPWMModulator


class TeslaSVPWMProfilleyici:
    """
    SVPWM Performans ve DC Bara Verim Profilleyicisi.
    """
    def __init__(self, sim_noktasi: int = 360):
        self.sim_noktasi = sim_noktasi

    def benchmark_svpwm(self) -> Dict[str, Any]:
        modulator = TeslaSVPWMModulator(v_dc_bus=400.0, switching_freq_hz=10000.0, dead_time_us=1.5)

        angles_deg = np.linspace(0, 360, self.sim_noktasi)
        v_ref_mag = 220.0  # Voltaj genliği

        sectors = []
        duty_a_list = []
        duty_b_list = []
        duty_c_list = []
        t1_list = []
        t2_list = []
        t0_list = []
        gecikmeler_us: List[float] = []

        for deg in angles_deg:
            rad = np.radians(deg)
            v_alpha = v_ref_mag * np.cos(rad)
            v_beta = v_ref_mag * np.sin(rad)

            t0 = time.perf_counter_ns()
            out = modulator.compute_phase_duty_cycles(v_alpha, v_beta)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

            sectors.append(out["sector"])
            duty_a_list.append(out["duty_a"])
            duty_b_list.append(out["duty_b"])
            duty_c_list.append(out["duty_c"])
            t1_list.append(out["t1_us"])
            t2_list.append(out["t2_us"])
            t0_list.append(out["t0_us"])

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        # SPWM vs SVPWM Maksimum Doğrusal Gerilim Kıyaslaması
        v_spwm_max = 400.0 / 2.0         # 200.0 V
        v_svpwm_max = 400.0 / np.sqrt(3) # 230.94 V
        dc_utilization_gain_pct = ((v_svpwm_max - v_spwm_max) / v_spwm_max) * 100.0

        return {
            "svpwm_step_ortalama_us": t_avg_us,
            "svpwm_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_svpwm_adimi": int(1e6 / max(t_avg_us, 1e-4)),
            "dc_gain_pct": dc_utilization_gain_pct,
            "v_spwm_max": v_spwm_max,
            "v_svpwm_max": v_svpwm_max,
            "angles_deg": angles_deg.tolist(),
            "sectors": sectors,
            "duty_a": duty_a_list,
            "duty_b": duty_b_list,
            "duty_c": duty_c_list,
            "t1_us": t1_list,
            "t2_us": t2_list,
            "t0_us": t0_list,
            "svpwm_gecikmeler": gecikmeler_us[:200]
        }
