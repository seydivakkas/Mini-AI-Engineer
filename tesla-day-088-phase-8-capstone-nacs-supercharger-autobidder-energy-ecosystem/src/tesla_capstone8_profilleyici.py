"""
Tesla Faz 8 Capstone Profilleyici Modülü
========================================
Bu modül; tüm Faz 8 enerji ve şarj mimarilerini içeren Capstone simülatörünün
çözümleme hızını ve RTOS gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_faz8_enerji_ekosistemi_simulatoru import TeslaPhase8EnergyEcosystemSimulator


class TeslaCapstone8Profilleyici:
    """
    Tesla Faz 8 Capstone Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_energy_ecosystem(self) -> Dict[str, Any]:
        sim = TeslaPhase8EnergyEcosystemSimulator()

        # 16 aracın farklı batarya dolulukları (%10 ile %85 arası)
        car_socs = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
                    55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 22.0]

        gecikmeler_us: List[float] = []

        for _ in range(self.iterations):
            s_inst = TeslaPhase8EnergyEcosystemSimulator()
            t0 = time.perf_counter_ns()
            _ = s_inst.step_ecosystem_simulation(
                grid_freq_hz=49.95,
                spot_price_usd_mwh=180.0,
                car_socs=car_socs,
                solar_irradiance_factor=0.9
            )
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        sim_res = sim.step_ecosystem_simulation(
            grid_freq_hz=49.95,
            spot_price_usd_mwh=180.0,
            car_socs=car_socs,
            solar_irradiance_factor=0.9
        )

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "step_ortalama_us": t_avg_us,
            "step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_dongu_kapasitesi": int(1e6 / max(t_avg_us, 1e-4)),
            "supercharger_load_kw": sim_res["supercharger_load_kw"],
            "solar_generated_kw": sim_res["solar_generated_kw"],
            "megapack_power_kw": sim_res["megapack_power_kw"],
            "net_grid_draw_kw": sim_res["net_grid_draw_kw"],
            "max_cable_temp": sim_res["max_cable_temp_c"],
            "grid_safety_ok": sim_res["grid_safety_ok"],
            "stall_powers": sim_res["stall_powers"],
            "gecikmeler": gecikmeler_us[:200]
        }
