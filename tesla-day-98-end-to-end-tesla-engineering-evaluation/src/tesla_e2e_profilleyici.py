"""
Tesla E2E Profilleyici Modülü
=============================
Bu modül; 8 temel mühendislik sütununun değerlendirilme ve kümülatif
skorlama hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_e2e_degerlendirici import TeslaE2EEngineeringEvaluator


class TeslaE2EProfilleyici:
    """
    Tesla Uçtan Uca Şampiyonluk Değerlendirme Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_evaluation_engine(self) -> Dict[str, Any]:
        evaluator = TeslaE2EEngineeringEvaluator()
        gecikmeler_us: List[float] = []

        for _ in range(self.iterations):
            e_inst = TeslaE2EEngineeringEvaluator()
            t0 = time.perf_counter_ns()
            pillars = e_inst.evaluate_all_pillars()
            _ = e_inst.calculate_championship_score(pillars)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        pillars = evaluator.evaluate_all_pillars()
        final_res = evaluator.calculate_championship_score(pillars)

        dizi = np.array(gecikmeler_us)

        return {
            "total_championship_score": final_res["total_championship_score"],
            "title_awarded": final_res["title_awarded"],
            "certification_status": final_res["certification_status"],
            "step_ortalama_us": float(np.mean(dizi)),
            "step_p99_us": float(np.percentile(dizi, 99)),
            "pillars_evaluated": len(pillars),
            "gecikmeler": gecikmeler_us[:200]
        }
