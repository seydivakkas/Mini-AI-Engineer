"""
Tesla 3D Render Profilleyici Modülü
===================================
Bu modül; 3D FSD Dünya Render motorunun MVP matris hesaplama hızını,
tepe noktası (vertex) izdüşüm süresini ve GPU 60 FPS bütçe uyumluluğunu profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_3d_render_motoru import Tesla3DWorldRenderer


class TeslaRenderProfilleyici:
    """
    Tesla 3D GPU Render Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_renderer(self) -> Dict[str, Any]:
        renderer = Tesla3DWorldRenderer()

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = renderer.render_fsd_scene()
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "render_step_ortalama_us": t_avg_us,
            "render_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_kare_kapasitesi": int(1e6 / max(t_avg_us, 1e-4)),
            "num_vertices": ciktilar["num_rendered_vertices"],
            "screen_res": ciktilar["screen_res"],
            "ego_pts": ciktilar["ego_screen_pts"],
            "left_lane": ciktilar["left_lane_screen"],
            "right_lane": ciktilar["right_lane_screen"],
            "path": ciktilar["path_screen"],
            "gecikmeler": gecikmeler_us[:200]
        }
