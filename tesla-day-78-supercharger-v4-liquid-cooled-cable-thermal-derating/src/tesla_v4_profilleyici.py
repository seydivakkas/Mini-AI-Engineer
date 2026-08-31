"""
Tesla Supercharger V4 Profilleyici Modülü
=========================================
Bu modül; Supercharger V4 kablo termal denklemi çözüm süresini ve
akım kısma (Derating) hesaplama hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_supercharger_v4_derater import TeslaSuperchargerV4CableDerater


class TeslaV4Profilleyici:
    """
    Tesla Supercharger V4 Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_v4_derating(self) -> Dict[str, Any]:
        derater = TeslaSuperchargerV4CableDerater()

        gecikmeler_us: List[float] = []

        for _ in range(self.iterations):
            d_inst = TeslaSuperchargerV4CableDerater()
            t0 = time.perf_counter_ns()
            _ = d_inst.get_derated_charging_current(78.5)
            _ = d_inst.step_thermal_model(demanded_current_a=500.0, dt=0.1)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        sim_derater = TeslaSuperchargerV4CableDerater()
        sim_res = sim_derater.simulate_charging_session(duration_s=120.0, demanded_current_a=500.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "step_ortalama_us": t_avg_us,
            "step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_kontrol_kapasitesi": int(1e6 / max(t_avg_us, 1e-4)),
            "final_temp_c": sim_res["final_temp_c"],
            "final_power_kw": sim_res["final_power_kw"],
            "zamanlar": sim_res["zamanlar_s"],
            "sicakliklar": sim_res["sicakliklar_c"],
            "gucler": sim_res["gucler_kw"],
            "akimlar": sim_res["akimlar_a"],
            "gecikmeler": gecikmeler_us[:200]
        }
