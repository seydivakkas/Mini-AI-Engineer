r"""
Tesla VectorLaneNet Profilleyici Modülü
=======================================
Bu modül; Şerit Polinomu hesaplama hızını, Analitik Eğrilik ($\kappa$)
türevini ve Kavşak Yönlendirilmiş Graf (DAG) sorgu gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_vector_lanenet_graf_topolojisi import TeslaVectorLaneNet


class TeslaLaneNetProfilleyici:
    """
    VectorLaneNet Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_vector_lanenet(self) -> Dict[str, Any]:
        lane_net = TeslaVectorLaneNet()
        graph = lane_net.construct_synthetic_intersection_graph()

        x_eval = np.linspace(0, 50, 100)
        poly_sample = np.array([-1.85, 0.02, 0.0005, 0.00001])

        gecikmeler_us: List[float] = []

        curvatures = []
        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            y_pts = lane_net.evaluate_lane_polynomial(poly_sample, x_eval)
            kappa_10 = lane_net.compute_lane_curvature(poly_sample, x_val=10.0)
            next_lanes = lane_net.get_legal_next_lanes(current_lane_id=0)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)
            curvatures.append(kappa_10)

        # Şerit Eğrilik Profili (0..50m)
        curv_profile = [lane_net.compute_lane_curvature(poly_sample, x) for x in x_eval]

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "lanenet_step_ortalama_us": t_avg_us,
            "lanenet_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_lanenet_adimi": int(1e6 / max(t_avg_us, 1e-4)),
            "kappa_10m": float(np.mean(curvatures)),
            "legal_next_lanes_0": next_lanes,
            "graph": graph,
            "x_eval": x_eval,
            "curv_profile": curv_profile,
            "poly_sample": poly_sample,
            "gecikmeler": gecikmeler_us[:200]
        }
