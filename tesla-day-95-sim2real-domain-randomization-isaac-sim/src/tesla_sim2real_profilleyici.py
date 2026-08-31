"""
Tesla Sim2Real Profilleyici Modülü
==================================
Bu modül; Isaac Sim Domain Randomization parametre örnekleme hızını ve
Sim2Real transfer kararlılığı değerlendirme gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_sim2real_randomizer import TeslaSim2RealDomainRandomizer


class TeslaSim2RealProfilleyici:
    """
    Tesla Sim2Real Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 50):
        self.iterations = iterations

    def benchmark_sim2real(self) -> Dict[str, Any]:
        randomizer = TeslaSim2RealDomainRandomizer()

        gecikmeler_us: List[float] = []

        for _ in range(self.iterations):
            r_inst = TeslaSim2RealDomainRandomizer()
            t0 = time.perf_counter_ns()
            _ = r_inst.sample_randomized_parameters()
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        eval_res = randomizer.evaluate_policy_robustness(num_episodes=100)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "num_episodes": eval_res["num_episodes"],
            "success_rate_pct": eval_res["success_rate_pct"],
            "average_reward": eval_res["average_reward"],
            "min_friction": eval_res["min_friction"],
            "max_latency_ms": eval_res["max_latency_ms"],
            "sim2real_ready": eval_res["sim2real_ready"],
            "step_ortalama_us": t_avg_us,
            "step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_ortam_ornekleme": int(1e6 / max(t_avg_us, 1e-4)),
            "rewards": eval_res["rewards"],
            "gecikmeler": gecikmeler_us[:200]
        }
