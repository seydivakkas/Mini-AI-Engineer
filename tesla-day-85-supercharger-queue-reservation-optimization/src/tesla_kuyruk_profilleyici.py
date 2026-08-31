"""
Tesla Supercharger Kuyruk Profilleyici Modülü
==============================================
Bu modül; M/M/c kuyruk analizi ve FSD rezervasyon karar süresini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_supercharger_kuyruk_yonetici import TeslaSuperchargerQueueManager


class TeslaKuyrukProfilleyici:
    """
    Tesla Supercharger Kuyruk Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_queue_optimization(self) -> Dict[str, Any]:
        mgr = TeslaSuperchargerQueueManager(num_stalls=12, service_rate_per_stall_per_hour=3.0)

        gecikmeler_us: List[float] = []

        for _ in range(self.iterations):
            m_inst = TeslaSuperchargerQueueManager()
            t0 = time.perf_counter_ns()
            _ = m_inst.calculate_mmc_metrics(arrival_rate_lambda=30.0)
            _ = m_inst.evaluate_fsd_route_reservation(current_arrival_rate=30.0, eta_minutes=18.0)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        # Farklı trafik yoğunlukları için Wq eğrisi
        lambdas = np.linspace(5.0, 34.0, 30)
        wq_curve = []
        rho_curve = []
        for lam in lambdas:
            res = mgr.calculate_mmc_metrics(lam)
            wq_curve.append(res["avg_wait_time_mins"])
            rho_curve.append(res["utilization_rho"])

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        opt_res = mgr.evaluate_fsd_route_reservation(current_arrival_rate=30.0, eta_minutes=15.0)

        return {
            "num_stalls": 12,
            "lambda_val": 30.0,
            "wait_mins": opt_res["current_station_wait_mins"],
            "decision": opt_res["decision"],
            "step_ortalama_us": t_avg_us,
            "step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_rezervasyon_kapasitesi": int(1e6 / max(t_avg_us, 1e-4)),
            "lambdas": list(lambdas),
            "wq_curve": wq_curve,
            "rho_curve": rho_curve,
            "gecikmeler": gecikmeler_us[:200]
        }
