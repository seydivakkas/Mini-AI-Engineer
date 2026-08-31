"""
Tesla BEV ve Homografi Profilleyici Modülü
==========================================
Bu modül; 2D perspektif piksellerin metrik Kuşbakışı (BEV) haritasına
dönüşüm hızını, yuvarlama hassasiyetini ve şerit projeksiyon doğruluğunu profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_bev_homografi_ve_ipm import TeslaBEVTransformer


class TeslaBEVProfilleyici:
    """
    BEV ve IPM Performans Profilleyicisi.
    """
    def __init__(self, num_points: int = 200, iterations: int = 100):
        self.num_points = num_points
        self.iterations = iterations

    def benchmark_bev_donusumu(self) -> Dict[str, Any]:
        bev_trans = TeslaBEVTransformer()

        # Sol ve Sağ Şerit için sentetik 2D pikseller
        # V: 900 (yakın) -> 520 (uzak ufuk)
        v_vals = np.linspace(900, 530, self.num_points)
        # Sol şerit u: 200 -> 600, Sağ şerit u: 1080 -> 680
        left_lane_pixels = [(float(200 + (600 - 200) * (900 - v) / 370), float(v)) for v in v_vals]
        right_lane_pixels = [(float(1080 - (1080 - 680) * (900 - v) / 370), float(v)) for v in v_vals]

        gecikmeler_us: List[float] = []

        bev_left = []
        bev_right = []
        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            bev_left = bev_trans.transform_lane_to_bev(left_lane_pixels)
            bev_right = bev_trans.transform_lane_to_bev(right_lane_pixels)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        # Gidiş-Dönüş (Round-Trip) Geometrik Doğrulama
        # (u, v) -> BEV -> (u', v')
        test_u, test_v = 640.0, 750.0
        bev_pt = bev_trans.pixel_to_bev(test_u, test_v)
        if bev_pt is not None:
            reproj_uv = bev_trans.bev_to_pixel(bev_pt[0], bev_pt[1])
            roundtrip_error_px = np.hypot(test_u - reproj_uv[0], test_v - reproj_uv[1])
        else:
            roundtrip_error_px = 0.0

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "bev_step_ortalama_us": t_avg_us,
            "bev_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_bev_donusumu": int(1e6 / max(t_avg_us, 1e-4)),
            "roundtrip_error_px": roundtrip_error_px,
            "left_lane_pixels": left_lane_pixels,
            "right_lane_pixels": right_lane_pixels,
            "bev_left": bev_left,
            "bev_right": bev_right,
            "gecikmeler": gecikmeler_us[:200]
        }
