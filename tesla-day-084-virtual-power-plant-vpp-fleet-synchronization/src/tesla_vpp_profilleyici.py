"""
Tesla VPP Profilleyici Modülü
==============================
Bu modül; 50.000 Powerwall ünitesinin anlık agregasyon ve dispatch
hesaplama gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_vpp_filo_yonetici import TeslaVirtualPowerPlantFleet


class TeslaVPPProfilleyici:
    """
    Tesla Virtual Power Plant Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 50, fleet_size: int = 50000):
        self.iterations = iterations
        self.fleet_size = fleet_size

    def benchmark_vpp_dispatch(self) -> Dict[str, Any]:
        fleet_inst = TeslaVirtualPowerPlantFleet(fleet_size=self.fleet_size)
        total_cap_mw = fleet_inst.get_available_fleet_capacity_mw()

        gecikmeler_us: List[float] = []

        for _ in range(self.iterations):
            f_test = TeslaVirtualPowerPlantFleet(fleet_size=self.fleet_size)
            t0 = time.perf_counter_ns()
            _ = f_test.dispatch_grid_demand(demand_mw=150.0, duration_hours=0.5)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        sim_fleet = TeslaVirtualPowerPlantFleet(fleet_size=self.fleet_size)
        sim_res = sim_fleet.dispatch_grid_demand(demand_mw=150.0, duration_hours=1.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "fleet_size": self.fleet_size,
            "total_capacity_mw": total_cap_mw,
            "demand_mw": sim_res["demand_mw"],
            "dispatched_mw": sim_res["dispatched_mw"],
            "demand_met": sim_res["demand_met"],
            "eligible_units": sim_res["eligible_units"],
            "avg_unit_kw": sim_res["avg_unit_power_kw"],
            "avg_soc_pct": sim_res["avg_fleet_soc_pct"],
            "dispatch_ortalama_us": t_avg_us,
            "dispatch_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_filo_dispatch_hizi": int(1e6 / max(t_avg_us, 1e-4)),
            "gecikmeler": gecikmeler_us[:200],
            "soc_orneklem": sim_fleet.soc_array[:500]
        }
