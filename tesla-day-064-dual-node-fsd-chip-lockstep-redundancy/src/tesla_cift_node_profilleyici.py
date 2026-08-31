"""
Tesla Çift Düğüm Profilleyici Modülü
====================================
Bu modül; FSD Node A / Node B oylama arabulucusu hızını,
karar ayrışması algılama süresini ve failover gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_cift_node_arabulucu import FSDHardwareArbiter


class TeslaCiftNodeProfilleyici:
    """
    FSD Çift Düğüm Arabulucu Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_arbiter(self) -> Dict[str, Any]:
        arbiter = FSDHardwareArbiter()

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = arbiter.arbitrate_decision(
                node_a_steer_rad=0.12,
                node_b_steer_rad=0.13,
                node_a_acc_mps2=0.8,
                node_b_acc_mps2=0.85,
                node_a_healthy=True,
                node_b_healthy=True
            )
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "arbiter_step_ortalama_us": t_avg_us,
            "arbiter_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_oylama_hacmi": int(1e6 / max(t_avg_us, 1e-4)),
            "mode": ciktilar["arbiter_mode"],
            "steer_applied": ciktilar["applied_steering_rad"],
            "acc_applied": ciktilar["applied_acc_mps2"],
            "steer_diff": ciktilar["steer_diff_rad"],
            "acc_diff": ciktilar["acc_diff_mps2"],
            "status_desc": ciktilar["status_desc"],
            "is_nominal": ciktilar["is_nominal"],
            "gecikmeler": gecikmeler_us[:200]
        }
