"""
Tesla Büyük Final Profilleyici Modülü
====================================
Bu modül; 99 günlük portföyün indeksleme, yönetici özeti çıkarma ve
sertifikasyon üretim hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_portfoy_gezgini import TeslaPortfolioNavigator


class TeslaFinalProfilleyici:
    """
    Tesla 99 Günlük Portföy Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_portfolio_indexing(self) -> Dict[str, Any]:
        nav = TeslaPortfolioNavigator()
        gecikmeler_us: List[float] = []

        for _ in range(self.iterations):
            n_inst = TeslaPortfolioNavigator()
            t0 = time.perf_counter_ns()
            _ = n_inst.get_weekly_curriculum()
            _ = n_inst.generate_executive_summary()
            _ = n_inst.generate_graduation_certificate()
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        exec_sum = nav.generate_executive_summary()
        cert = nav.generate_graduation_certificate()
        dizi = np.array(gecikmeler_us)

        return {
            "total_days_completed": exec_sum["total_days_completed"],
            "total_weeks_completed": exec_sum["total_weeks_completed"],
            "total_test_pass_rate_pct": exec_sum["total_test_pass_rate_pct"],
            "degree_awarded": cert["degree_awarded"],
            "honors": cert["honors"],
            "step_ortalama_us": float(np.mean(dizi)),
            "step_p99_us": float(np.percentile(dizi, 99)),
            "gecikmeler": gecikmeler_us[:200]
        }
