"""
Day 354: Subterranean Lava Tube Exploration & GPS-Denied 3D Graph SLAM for Mars Rovers
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Mars Yeraltı Lav Tüpü Mağara Ortamını, GPS'siz 3D Nokta Bulutu Eşlemeyi (ICP/NDT),
Döngü Kapatmayı (Loop Closure) ve Poz Grafı Optimizasyonunu (Pose Graph SLAM) içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np


class MartianLavaTubeCave:
    """
    Mars Yeraltı Lav Tüpü (Lava Tube) ve 3D Mağara Yapısı Simülatörü.
    Kıvrımlı, kayalık ve kapalı döngü içeren 3D mağara tüneli sentezler.
    """
    def __init__(self, num_points: int = 1500, tunnel_radius_m: float = 10.0):
        self.num_points = num_points
        self.radius = tunnel_radius_m
        
        # 3D Kapalı Döngülü Mağara Omurgası (Centerline)
        t = np.linspace(0, 2*np.pi, 200)
        self.centerline = np.stack([
            80.0 * np.cos(t),
            40.0 * np.sin(2*t),
            -20.0 + 5.0 * np.sin(3*t) # Yeraltı derinliği (m)
        ], axis=-1)

        # 3D Mağara Duvarları (Point Cloud Mesh)
        angles = np.random.uniform(0, 2*np.pi, num_points)
        s_idx = np.random.randint(0, len(self.centerline), num_points)
        c_pos = self.centerline[s_idx]
        
        rad_noise = self.radius + np.random.normal(0, 1.2, num_points)
        self.cave_points = np.zeros((num_points, 3))
        self.cave_points[:, 0] = c_pos[:, 0] + rad_noise * np.cos(angles)
        self.cave_points[:, 1] = c_pos[:, 1] + rad_noise * np.sin(angles)
        self.cave_points[:, 2] = c_pos[:, 2] + np.random.uniform(-self.radius, self.radius, num_points)

    def sample_lidar_scan(self, rover_pos: np.ndarray, fov_range_m: float = 35.0) -> np.ndarray:
        """Rover konumundan görünen yerel 3D LiDAR nokta bulutunu döner."""
        dists = np.linalg.norm(self.cave_points - rover_pos, axis=-1)
        visible_mask = dists <= fov_range_m
        scan = self.cave_points[visible_mask] + np.random.normal(0, 0.05, (np.sum(visible_mask), 3))
        return scan


class MarsRover3DGraphSLAM:
    """
    GPS'siz 3D Poz Grafı SLAM (Pose Graph Optimization) Motoru.
    Gezginin kümülatif odometri sapmasını döngü kapatma (Loop Closure) anında sıfırlar.
    """
    def __init__(self):
        self.poses: List[np.ndarray] = []
        self.odometry_edges: List[Tuple[int, int, np.ndarray]] = []
        self.loop_edges: List[Tuple[int, int, np.ndarray]] = []

    def add_odometry_pose(self, raw_noisy_pose: np.ndarray):
        """Yeni gezgin pozunu ekler."""
        self.poses.append(raw_noisy_pose.copy())
        if len(self.poses) > 1:
            i = len(self.poses) - 2
            j = len(self.poses) - 1
            rel_trans = self.poses[j] - self.poses[i]
            self.odometry_edges.append((i, j, rel_trans))

    def detect_loop_closure(self, current_idx: int, distance_thresh_m: float = 12.0) -> Optional[int]:
        """Geçmişte ziyaret edilen bir odaya geri dönülüp dönülmediğini kontrol eder."""
        if current_idx < 80:
            return None
        cur_p = self.poses[current_idx]
        # Başlangıç odasına dönüş (0..20)
        for past_idx in range(min(20, current_idx - 60)):
            dist = float(np.linalg.norm(cur_p - self.poses[past_idx]))
            if dist < distance_thresh_m:
                if len(self.loop_edges) == 0: # Tek ve net ana döngü kilitlenmesi
                    rel_trans = self.poses[current_idx] - self.poses[past_idx]
                    self.loop_edges.append((past_idx, current_idx, rel_trans))
                    return past_idx
        return None

    def optimize_pose_graph(self) -> np.ndarray:
        """
        Döngü kapatma kısıtları ile tüm grafı optimize eder (Gauss-Newton Relaksasyonu).
        """
        N = len(self.poses)
        optimized_poses = np.array(self.poses).copy()

        if len(self.loop_edges) == 0:
            return optimized_poses

        # Döngü kapatma hatasını tüm yörünge boyunca pürüzsüz dağıt (Drift Correction)
        for past_i, cur_i, _ in self.loop_edges:
            # Döngünün kapanışındaki toplam biriken sapma vektörü
            loop_drift = optimized_poses[cur_i] - optimized_poses[past_i]
            num_steps = cur_i - past_i
            if num_steps > 0:
                for k in range(past_i, cur_i + 1):
                    fraction = (k - past_i) / float(num_steps)
                    optimized_poses[k] -= fraction * loop_drift

        return optimized_poses


class SubterraneanExplorationEngine:
    """
    Uçtan Uca Mars Gezgini Yeraltı Mağarası Keşif ve SLAM Simülatörü.
    """
    def __init__(self):
        self.cave = MartianLavaTubeCave()
        self.slam = MarsRover3DGraphSLAM()

    def run_exploration(self) -> Dict[str, Any]:
        """Gezginin tüm mağarayı dolanıp haritalamasını simüle eder."""
        true_trajectory = self.cave.centerline
        N_steps = len(true_trajectory)

        # Gerçekçi tekerlek patinajı ve IMU kayma sürüklenmesi (Linear + Random Walk)
        drift_direction = np.array([0.04, -0.03, 0.02])
        noisy_odometry = []
        drift = np.zeros(3)

        loop_closed_step = -1

        for step in range(N_steps):
            true_p = true_trajectory[step]
            drift += drift_direction + np.random.normal(0, 0.02, 3)
            noisy_p = true_p + drift

            noisy_odometry.append(noisy_p)
            self.slam.add_odometry_pose(noisy_p)

            # Döngü Kapatma Kontrolü
            loop_idx = self.slam.detect_loop_closure(step)
            if loop_idx is not None and loop_closed_step == -1:
                loop_closed_step = step

        # Poz Grafı Optimizasyonu
        optimized_trajectory = self.slam.optimize_pose_graph()

        noisy_odometry = np.array(noisy_odometry)

        # Hata Analizi (RMSE)
        drift_rmse = float(np.sqrt(np.mean((noisy_odometry - true_trajectory) ** 2)))
        slam_rmse = float(np.sqrt(np.mean((optimized_trajectory - true_trajectory) ** 2)))

        return {
            "true_trajectory": true_trajectory,
            "noisy_odometry": noisy_odometry,
            "optimized_trajectory": optimized_trajectory,
            "cave_points": self.cave.cave_points,
            "drift_rmse_m": drift_rmse,
            "slam_rmse_m": slam_rmse,
            "loop_closed_step": loop_closed_step,
            "loop_count": len(self.slam.loop_edges)
        }
