"""
Tesla Batarya SoH ve Yaşlanma Profilleyici Modülü
=================================================
Bu modül; 2000 döngülük hücre ömrü simülasyonunu, RLS çevrimiçi iç direnç
takibini ve Supercharger vs Ev tipi şarj yaşlanma farkını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_soh_ve_ic_direnc_izleyici import (
    BatteryCycleAgingSimulator,
    RecursiveLeastSquaresR0,
    calculate_soh_capacity,
    calculate_soh_resistance
)


class TeslaSoHProfilleyici:
    """
    Batarya Yaşlanma ve SoH Performans Profilleyicisi.
    """
    def __init__(self, max_cycles: int = 2000):
        self.max_cycles = max_cycles

    def benchmark_batarya_soh(self) -> Dict[str, Any]:
        # 1. Normal Şarj (25°C, 0.5C) vs Supercharger (45°C, 2.5C) Yaşlanma Kıyaslaması
        sim_normal = BatteryCycleAgingSimulator(fresh_capacity_ah=75.0, fresh_r0_ohm=0.0015)
        sim_fast = BatteryCycleAgingSimulator(fresh_capacity_ah=75.0, fresh_r0_ohm=0.0015)

        cycles_list = []
        normal_soh_list = []
        fast_soh_list = []
        normal_r0_list = []
        fast_r0_list = []

        step_size = 50
        for c in range(0, self.max_cycles + 1, step_size):
            cycles_list.append(c)
            if c > 0:
                sim_normal.step_cycles(cycle_count=step_size, temp_c=25.0, dod_depth_of_discharge=0.70)
                sim_fast.step_cycles(cycle_count=step_size, temp_c=45.0, dod_depth_of_discharge=0.90)

            st_norm = sim_normal.get_health_status()
            st_fast = sim_fast.get_health_status()

            normal_soh_list.append(st_norm["soh_capacity_pct"])
            fast_soh_list.append(st_fast["soh_capacity_pct"])
            normal_r0_list.append(st_norm["r0_ohm"] * 1000.0)  # mOhm
            fast_r0_list.append(st_fast["r0_ohm"] * 1000.0)

        # 2. RLS Çevrimiçi İç Direnç Kestirim Doğruluğu
        rls = RecursiveLeastSquaresR0(initial_r0_guess=0.0012, lambda_forgetting=0.995)
        true_r0 = 0.0022  # Yaşlanmış hücre (2.2 mOhm)
        gecikmeler_rls_us: List[float] = []
        rls_tahminler = []

        for i in range(500):
            delta_i = float(40.0 + 20.0 * np.sin(i * 0.1))
            delta_v = delta_i * true_r0 + np.random.normal(0, 0.001)

            t0 = time.perf_counter_ns()
            tahmin_r0 = rls.update(delta_i, delta_v)
            t1 = time.perf_counter_ns()
            gecikmeler_rls_us.append(float(t1 - t0) / 1000.0)
            rls_tahminler.append(tahmin_r0 * 1000.0)

        rls_dizi = np.array(gecikmeler_rls_us)
        t_rls_avg_us = float(np.mean(rls_dizi))

        return {
            "rls_step_ortalama_us": t_rls_avg_us,
            "rls_step_p99_us": float(np.percentile(rls_dizi, 99)),
            "saniyelik_rls_adimi": int(1e6 / max(t_rls_avg_us, 1e-4)),
            "final_soh_normal_pct": normal_soh_list[-1],
            "final_soh_fast_pct": fast_soh_list[-1],
            "final_r0_normal_mohm": normal_r0_list[-1],
            "final_r0_fast_mohm": fast_r0_list[-1],
            "cycles": cycles_list,
            "normal_soh": normal_soh_list,
            "fast_soh": fast_soh_list,
            "normal_r0": normal_r0_list,
            "fast_r0": fast_r0_list,
            "rls_tahminler": rls_tahminler,
            "true_r0_mohm": true_r0 * 1000.0
        }
