"""
Tesla Görsel Odometri Profilleyici Modülü
==========================================
Bu modül; PnP + RANSAC kamera poz kestirim gecikmesini, yeniden izdüşüm
hatasını (Reprojection Error) ve kapalı döngü (Loop Closure) hassasiyetini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_gorsel_odometri_ve_slam import TeslaVisualOdometrySLAM


class TeslaVOProfilleyici:
    """
    Görsel Odometri ve Semantik SLAM Performans Profilleyicisi.
    """
    def __init__(self, num_points: int = 150):
        self.num_points = num_points

    def benchmark_vo_and_slam(self) -> Dict[str, Any]:
        slam = TeslaVisualOdometrySLAM()

        # 3D Statik Çevre Noktaları (Yol kenarı binalar ve direkler)
        np.random.seed(42)
        pts_3d = np.random.uniform(-20, 20, (self.num_points, 3))
        pts_3d[:, 2] = np.random.uniform(5, 40, self.num_points)  # Z derinlik 5-40m

        # Kamera Projeksiyonu
        R_true = np.eye(3)
        t_true = np.array([[0.2], [0.0], [1.5]])  # 1.5m ileri hareket
        pts_2d = np.zeros((self.num_points, 2))
        for i in range(self.num_points):
            pts_2d[i] = slam.project_3d_to_2d(pts_3d[i], R_true, t_true)

        # Semantik etiketler: %15 dinamik araç/yaya nesnesi
        semantics = np.zeros(self.num_points, dtype=int)
        semantics[np.random.choice(self.num_points, size=22, replace=False)] = 1

        gecikmeler_us: List[float] = []

        best_R, best_t, inliers, mean_err = None, None, 0, 0.0
        for _ in range(50):
            t0 = time.perf_counter_ns()
            best_R, best_t, inliers, mean_err = slam.estimate_pose_pnp_ransac(
                pts_3d, pts_2d, semantic_labels=semantics, max_iters=40
            )
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        # Kapalı Döngü Simülasyonu (30m x 30m Dikdörtgen Rota)
        trajectory_x = []
        trajectory_z = []
        loop_closed = False

        # Kare rota koordinatları
        steps_per_side = 8
        side_len = 16.0
        cur_pos = np.array([[0.0], [0.0], [0.0]])
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        for d in directions:
            for _ in range(steps_per_side):
                cur_pos[0, 0] += d[0] * (side_len / steps_per_side)
                cur_pos[2, 0] += d[1] * (side_len / steps_per_side)
                is_kf, is_l = slam.check_keyframe_and_loop_closure(cur_pos)
                if is_l:
                    loop_closed = True
                trajectory_x.append(cur_pos[0, 0])
                trajectory_z.append(cur_pos[2, 0])

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "pnp_step_ortalama_us": t_avg_us,
            "pnp_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_vo_adimi": int(1e6 / max(t_avg_us, 1e-4)),
            "inlier_orani_pct": float(inliers / max(self.num_points - 22, 1) * 100.0),
            "reproj_error_px": float(mean_err),
            "keyframes_count": len(slam.keyframes),
            "loop_closed": loop_closed,
            "traj_x": trajectory_x,
            "traj_z": trajectory_z,
            "pts_3d": pts_3d,
            "pts_2d": pts_2d,
            "gecikmeler": gecikmeler_us[:200]
        }
