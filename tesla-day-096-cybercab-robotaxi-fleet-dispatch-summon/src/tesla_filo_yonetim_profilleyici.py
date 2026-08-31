"""
Tesla Cybercab Filo Yönetim Profilleyici Modülü
===============================================
Bu modül; Cybercab otonom çağırma ve filo görevlendirme eşleştirme hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_cybercab_filo_yoneticisi import TeslaCybercabFleetDispatcher, CybercabVehicle, PassengerRequest


class TeslaFiloYonetimProfilleyici:
    """
    Tesla Cybercab Filo Yönetimi Performans Profilleyicisi.
    """
    def __init__(self, fleet_size: int = 100):
        self.size = fleet_size

    def generate_synthetic_fleet(self) -> List[CybercabVehicle]:
        fleet = []
        for i in range(self.size):
            # 10x10 km şehir alanı
            x = float(np.random.uniform(0.0, 10.0))
            y = float(np.random.uniform(0.0, 10.0))
            soc = float(np.random.uniform(15.0, 95.0))
            stat = "AVAILABLE" if soc >= 20.0 and np.random.rand() > 0.3 else "ON_TRIP"
            fleet.append(CybercabVehicle(
                cab_id=f"CAB_{i:04d}",
                x_km=x,
                y_km=y,
                soc_pct=soc,
                status=stat
            ))
        return fleet

    def benchmark_fleet_dispatch(self) -> Dict[str, Any]:
        dispatcher = TeslaCybercabFleetDispatcher()
        fleet = self.generate_synthetic_fleet()

        req = PassengerRequest(
            req_id="REQ_999",
            pickup_x_km=5.0,
            pickup_y_km=5.0,
            dest_x_km=8.5,
            dest_y_km=2.0
        )

        gecikmeler_us: List[float] = []

        for _ in range(50):
            d_inst = TeslaCybercabFleetDispatcher()
            t0 = time.perf_counter_ns()
            _ = d_inst.dispatch_trip(req, fleet)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dispatch_res = dispatcher.dispatch_trip(req, fleet)
        charged_cabs = dispatcher.auto_supercharge_rebalancing(fleet)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "fleet_size": self.size,
            "assigned_cab_id": dispatch_res["assigned_cab_id"],
            "pickup_distance_km": dispatch_res["pickup_distance_km"],
            "eta_minutes": dispatch_res["eta_minutes"],
            "auto_charged_count": len(charged_cabs),
            "step_ortalama_us": t_avg_us,
            "step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_eslestirme_hizi": int(1e6 / max(t_avg_us, 1e-4)),
            "gecikmeler": gecikmeler_us[:200]
        }
