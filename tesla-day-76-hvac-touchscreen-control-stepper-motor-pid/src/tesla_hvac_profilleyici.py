"""
Tesla HVAC Profilleyici Modülü
==============================
Bu modül; HVAC PID hesaplama süresini, step motor darbe oluşturma hızını
ve kapalı döngü termal simülasyon gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_hvac_pid_kontrolcu import TeslaHVACPIDController


class TeslaHVACProfilleyici:
    """
    Tesla HVAC PID Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_hvac_pid(self) -> Dict[str, Any]:
        controller = TeslaHVACPIDController(dt=0.1)

        gecikmeler_us: List[float] = []

        for _ in range(self.iterations):
            ctrl_inst = TeslaHVACPIDController(dt=0.1)
            t0 = time.perf_counter_ns()
            _ = ctrl_inst.step()
            _ = ctrl_inst.calculate_stepper_pulses(25.0)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        sim_ctrl = TeslaHVACPIDController(dt=0.1, initial_temp_c=35.0, target_temp_c=21.5)
        sim_res = sim_ctrl.simulate_cooling_trajectory(duration_s=60.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "hvac_step_ortalama_us": t_avg_us,
            "hvac_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_pid_dongusu": int(1e6 / max(t_avg_us, 1e-4)),
            "final_temp_c": sim_res["final_temp_c"],
            "settling_achieved": sim_res["settling_achieved"],
            "zamanlar": sim_res["zamanlar_s"],
            "sicakliklar": sim_res["sicakliklar_c"],
            "gucler": sim_res["gucler_pct"],
            "gecikmeler": gecikmeler_us[:200]
        }
