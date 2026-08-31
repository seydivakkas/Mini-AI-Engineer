"""
Tesla Dojo Profilleyici Modülü
==============================
Bu modül; Dojo D1 NoC yönlendirme algoritmasının hesaplama hızını ve
Training Tile içi paket transfer gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_dojo_d1_mesh_yonlendirici import TeslaDojoMeshRouter


class TeslaDojoProfilleyici:
    """
    Tesla Dojo Süperbilgisayar NoC Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_dojo_routing(self) -> Dict[str, Any]:
        router = TeslaDojoMeshRouter(grid_width=5, grid_height=5)

        gecikmeler_us: List[float] = []

        for _ in range(self.iterations):
            r_inst = TeslaDojoMeshRouter()
            t0 = time.perf_counter_ns()
            _ = r_inst.calculate_packet_transfer_latency(src=(0, 0), dst=(4, 4), payload_bytes=1024 * 1024)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        # Tüm 25 çip arası ortalama gecikme matrisi (5x5)
        hop_matrix = np.zeros((5, 5))
        for x in range(5):
            for y in range(5):
                hop_matrix[x, y] = router.compute_manhattan_distance((0, 0), (x, y))

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        diag_res = router.calculate_packet_transfer_latency(src=(0, 0), dst=(4, 4), payload_bytes=1024 * 1024)

        return {
            "num_chips": router.num_chips,
            "tile_pflops": router.tile_pflops,
            "hop_count_corner_to_corner": diag_res["hops"],
            "total_latency_ns": diag_res["total_latency_ns"],
            "effective_bw_gb_s": diag_res["effective_bw_gb_s"],
            "step_ortalama_us": t_avg_us,
            "step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_yonlendirme_kapasitesi": int(1e6 / max(t_avg_us, 1e-4)),
            "hop_matrix": hop_matrix,
            "path_sample": diag_res["path"],
            "gecikmeler": gecikmeler_us[:200]
        }
