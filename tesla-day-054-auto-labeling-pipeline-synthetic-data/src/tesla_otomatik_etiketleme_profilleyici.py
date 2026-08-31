"""
Tesla Otomatik Etiketleme Profilleyici Modülü
=============================================
Bu modül; Çift Yönlü Yörünge Düzeltme (Bidirectional Smoothing) hızını,
Çoklu Sürüş Hizalama doğruluğunu ve 3D IoU kalite metriklerini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_otomatik_etiketleme_ve_sentetik_veri import TeslaAutoLabelingPipeline


class TeslaOtomatikEtiketlemeProfilleyici:
    """
    Otomatik Etiketleme ve Sentetik Veri Performans Profilleyicisi.
    """
    def __init__(self, trajectory_len: int = 100, iterations: int = 100):
        self.trajectory_len = trajectory_len
        self.iterations = iterations

    def benchmark_auto_labeling_pipeline(self) -> Dict[str, Any]:
        pipeline = TeslaAutoLabelingPipeline(time_steps=self.trajectory_len)

        # Gürültülü Sentetik Yörünge
        t = np.linspace(0, 10, self.trajectory_len)
        true_traj = np.column_stack([np.sin(t) * 2.0, t * 10.0])
        np.random.seed(42)
        noisy_traj = true_traj + np.random.normal(0, 0.3, true_traj.shape)

        # Çoklu Sürüş Noktaları
        trip1 = np.random.normal(0, 5, (500, 3))
        trip2 = np.random.normal(0, 5, (500, 3))

        gecikmeler_us: List[float] = []
        smoothed = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            smoothed = pipeline.bidirectional_temporal_smoothing(noisy_traj)
            align_res = pipeline.align_multi_trip_point_clouds(trip1, trip2)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        # 3D BBox IoU Doğrulaması
        bbox_pred = np.array([0.0, 15.0, 0.0, 2.0, 4.5, 1.5])
        bbox_gt = np.array([0.05, 15.02, 0.02, 2.0, 4.5, 1.5])
        iou_val = pipeline.calculate_3d_bbox_iou(bbox_pred, bbox_gt)

        # Hata Azaltımı (RMSE)
        rmse_noisy = float(np.sqrt(np.mean((noisy_traj - true_traj) ** 2)))
        rmse_smooth = float(np.sqrt(np.mean((smoothed - true_traj) ** 2)))
        noise_reduction_pct = ((rmse_noisy - rmse_smooth) / rmse_noisy) * 100.0

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "autolabel_step_ortalama_us": t_avg_us,
            "autolabel_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_klip_karesi": int(1e6 / max(t_avg_us, 1e-4)),
            "3d_bbox_iou": iou_val,
            "rmse_noisy": rmse_noisy,
            "rmse_smooth": rmse_smooth,
            "noise_reduction_pct": noise_reduction_pct,
            "alignment_rmse_cm": align_res["alignment_rmse_cm"],
            "total_points": align_res["total_points"],
            "true_traj": true_traj,
            "noisy_traj": noisy_traj,
            "smoothed_traj": smoothed,
            "gecikmeler": gecikmeler_us[:200]
        }
