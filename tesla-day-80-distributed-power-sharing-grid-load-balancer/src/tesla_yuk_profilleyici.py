"""
Tesla Yük Dengeleyici Profilleyici Modülü
=========================================
Bu modül; 8 stall'luk Supercharger istasyonu dinamik güç paylaştırma
süresini ve trafo aşırı yük koruma kontrol gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_dinamik_yuk_dengeleyici import TeslaDynamicLoadBalancer


class TeslaYukProfilleyici:
    """
    Tesla Dinamik Yük Dengeleme Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_load_balancing(self) -> Dict[str, Any]:
        balancer = TeslaDynamicLoadBalancer(grid_capacity_kw=1000.0, max_stall_power_kw=250.0)
        soc_fleet = [12.0, 25.0, 38.0, 55.0, 70.0, 82.0, 88.0, 92.0]

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            b_inst = TeslaDynamicLoadBalancer(grid_capacity_kw=1000.0, max_stall_power_kw=250.0)
            t0 = time.perf_counter_ns()
            ciktilar = b_inst.balance_power(soc_fleet)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "balance_step_ortalama_us": t_avg_us,
            "balance_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_dengeleme_kapasitesi": int(1e6 / max(t_avg_us, 1e-4)),
            "allocated_powers": ciktilar["allocated_powers_kw"],
            "total_allocated": ciktilar["total_allocated_kw"],
            "grid_headroom": ciktilar["grid_headroom_kw"],
            "overload_prevented": ciktilar["overload_prevented"],
            "gecikmeler": gecikmeler_us[:200]
        }
