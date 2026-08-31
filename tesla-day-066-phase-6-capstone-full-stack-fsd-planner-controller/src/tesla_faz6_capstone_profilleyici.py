"""
Tesla Faz 6 Capstone Profilleyici Modülü
========================================
Bu modül; Full-Stack FSD Planlayıcı ve MPC Kontrolcü motorunun uçtan uca
çözüm süresini, ASIL-D döngü hızını ve RTOS çalışma frekansını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_faz6_capstone_planner_controller import TeslaFullStackFSDPlannerController


class TeslaFaz6CapstoneProfilleyici:
    """
    Tesla Faz 6 Capstone Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_pipeline(self) -> Dict[str, Any]:
        engine = TeslaFullStackFSDPlannerController()

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = engine.run_full_fsd_pipeline()
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "capstone_step_ortalama_us": t_avg_us,
            "capstone_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_fsd_döngüsü": int(1e6 / max(t_avg_us, 1e-4)),
            "ciktilar": ciktilar,
            "gecikmeler": gecikmeler_us[:200]
        }
