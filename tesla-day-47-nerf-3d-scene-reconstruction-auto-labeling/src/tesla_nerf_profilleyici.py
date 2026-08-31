"""
Tesla NeRF Profilleyici Modülü
==============================
Bu modül; Hacimsel Işın İzleme (Volume Rendering) hesaplama hızını,
derinlik doğruluğunu ve otomatik etiketleme (Auto-Labeling) PSNR metriklerini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_nerf_ve_otomatik_etiketleme import TeslaNeRFAutoLabeler


class TeslaNeRFProfilleyici:
    """
    NeRF ve Otomatik Etiketleme Performans Profilleyicisi.
    """
    def __init__(self, num_rays: int = 100):
        self.num_rays = num_rays

    def benchmark_nerf_auto_labeling(self) -> Dict[str, Any]:
        labeler = TeslaNeRFAutoLabeler(num_samples_per_ray=32, near_m=1.0, far_m=35.0)

        # Kamera Görüş Açısı Işın Demeti (X ekseninde açı taraması)
        ray_origins = np.zeros((self.num_rays, 3))
        ray_dirs = np.zeros((self.num_rays, 3))
        angles = np.linspace(-0.2, 0.2, self.num_rays)
        for i, ang in enumerate(angles):
            ray_dirs[i] = np.array([np.sin(ang), np.cos(ang), 0.0])

        rendered_depths = np.zeros(self.num_rays)
        gecikmeler_us: List[float] = []

        for i in range(self.num_rays):
            t0 = time.perf_counter_ns()
            rgb, d, op = labeler.render_volume_ray(ray_origins[i], ray_dirs[i])
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)
            rendered_depths[i] = d

        # Otomatik 3D Bounding Box Etiketleme
        bbox_res = labeler.auto_label_3d_bounding_box(rendered_depths, ray_origins, ray_dirs)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "nerf_ray_ortalama_us": t_avg_us,
            "nerf_ray_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_nerf_isini": int(1e6 / max(t_avg_us, 1e-4)),
            "psnr_db": bbox_res["psnr_db"],
            "bbox_center": bbox_res["bbox_center"],
            "bbox_dims": bbox_res["dimensions"],
            "point_count": bbox_res["point_count"],
            "rendered_depths": rendered_depths,
            "angles": angles,
            "gecikmeler": gecikmeler_us[:200]
        }
