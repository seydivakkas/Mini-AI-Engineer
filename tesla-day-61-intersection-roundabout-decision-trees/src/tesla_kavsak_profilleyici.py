"""
Tesla Kavşak Profilleyici Modülü
================================
Bu modül; Döner kavşak karar ağacı çalışma süresini, çoklu araç TTC hesaplama
gecikmesini ve sonlu durum makinesi (FSM) geçiş performansını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_kavsak_karar_agaci import TeslaIntersectionDecisionTree


class TeslaKavsakProfilleyici:
    """
    Kavşak ve Döner Kavşak Karar Motoru Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_decision_tree(self) -> Dict[str, Any]:
        tree = TeslaIntersectionDecisionTree(min_ttc_safe_s=3.5)

        test_vehicles = [
            {"id": 1, "dist_m": 45.0, "speed_mps": 10.0},  # TTC = 4.5s (Güvenli)
            {"id": 2, "dist_m": 60.0, "speed_mps": 12.0},  # TTC = 5.0s
            {"id": 3, "dist_m": 25.0, "speed_mps": 10.0}   # TTC = 2.5s (Kritik)
        ]

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = tree.evaluate_roundabout_scenario(ego_dist_to_yield_line_m=5.0, circulating_vehicles=test_vehicles)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "decision_step_ortalama_us": t_avg_us,
            "decision_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_karar_cevrimi": int(1e6 / max(t_avg_us, 1e-4)),
            "state": ciktilar["state"],
            "action": ciktilar["action"],
            "min_ttc_s": ciktilar["min_ttc_s"],
            "critical_id": ciktilar["critical_vehicle_id"],
            "can_enter": ciktilar["can_enter"],
            "target_acc": ciktilar["target_acc_mps2"],
            "gecikmeler": gecikmeler_us[:200]
        }
