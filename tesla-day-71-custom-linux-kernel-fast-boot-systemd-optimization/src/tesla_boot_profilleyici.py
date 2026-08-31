"""
Tesla Fast-Boot Profilleyici Modülü
===================================
Bu modül; boot sürelerini, systemd blame analiz hızını ve
<2.0 saniye Fast-Boot hedef uyumluluğunu profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_fast_boot_yonetici import TeslaFastBootOptimizer


class TeslaBootProfilleyici:
    """
    Tesla Fast-Boot Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_boot_analyzer(self) -> Dict[str, Any]:
        opt = TeslaFastBootOptimizer(target_boot_limit_s=2.0)

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = opt.optimize_systemd_chain()
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "analyzer_step_ortalama_us": t_avg_us,
            "analyzer_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_analiz_hacmi": int(1e6 / max(t_avg_us, 1e-4)),
            "boot_stages": ciktilar["boot_stages"],
            "raw_services": ciktilar["raw_services"],
            "opt_services": ciktilar["optimized_services"],
            "slow_before": ciktilar["slow_services_before"],
            "slow_after": ciktilar["slow_services_after"],
            "is_compliant": ciktilar["is_fast_boot_compliant"],
            "gecikmeler": gecikmeler_us[:200]
        }
