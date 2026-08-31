"""
Tesla BESS Profilleyici Modülü
==============================
Bu modül; Megapack P-f ve Q-V Droop kontrol tepki süresini ve
saniyelik şebeke frekans düzeltme kapasitesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_megapack_bess_kontrolcu import TeslaMegapackBESSController


class TeslaBESSProfilleyici:
    """
    Tesla Megapack BESS Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_bess_droop(self) -> Dict[str, Any]:
        bess = TeslaMegapackBESSController()

        gecikmeler_us: List[float] = []

        for _ in range(self.iterations):
            b_inst = TeslaMegapackBESSController()
            t0 = time.perf_counter_ns()
            _ = b_inst.compute_active_droop_power(49.85)
            _ = b_inst.compute_reactive_droop_power(395.0)
            _ = b_inst.step_bess_simulation(grid_freq_hz=49.85, grid_voltage_v=395.0, dt_s=0.1)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        # 60 saniyelik frekans dalgalanması simülasyonu
        sim_bess = TeslaMegapackBESSController()
        t_arr = np.linspace(0, 60, 120)
        # 49.8 Hz ile 50.2 Hz arası dalgalanan şebeke
        freq_wave = 50.0 + 0.2 * np.sin(2 * np.pi * 0.05 * t_arr)
        p_history = []
        soc_history = []

        for f_val in freq_wave:
            res = sim_bess.step_bess_simulation(grid_freq_hz=f_val, dt_s=0.5)
            p_history.append(res["active_power_kw"])
            soc_history.append(res["soc_pct"])

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "step_ortalama_us": t_avg_us,
            "step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_droop_kapasitesi": int(1e6 / max(t_avg_us, 1e-4)),
            "zamanlar": t_arr,
            "frekanslar": freq_wave,
            "gucler": p_history,
            "soclar": soc_history,
            "final_soc": soc_history[-1],
            "gecikmeler": gecikmeler_us[:200]
        }
