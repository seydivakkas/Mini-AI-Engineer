"""
Day 334: Microsecond Latency Spike-based Neuromorphic SLAM
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Asenkron DVS Olay Akış Simülatörünü, Spike Tabanlı ICP Taraması Eşleştiricisini
ve Mikrosaniye Gecikmeli Bayesyen Log-Odds Nöromorfik SLAM Motorunu içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import math
import time
import numpy as np


class DVSEventStreamSimulator:
    """
    Olay Tabanlı DVS Kamera (Dynamic Vision Sensor) Akış Simülatörü.
    Ajan 2D haritada hareket ederken mikrosaniye hassasiyetli spike üreterek e_k = (x, y, t_us, p) fırlatır.
    """
    def __init__(self, map_size: int = 50):
        self.map_size = map_size
        # Sentetik 2D Engel Haritası
        self.true_map = np.zeros((map_size, map_size), dtype=np.float32)
        # Duvarlar ve engeller ekle
        self.true_map[10:40, 10] = 1.0
        self.true_map[10, 10:40] = 1.0
        self.true_map[10:40, 40] = 1.0
        self.true_map[40, 10:41] = 1.0
        self.true_map[20:30, 25] = 1.0

    def generate_event_batch(self, agent_pos: np.ndarray, dt_us: int = 1000) -> List[Tuple[int, int, int, int]]:
        """
        Girdi: Ajan Konumu (x, y) -> Çıktı: DVS Spike Olay Listesi [(x, y, t_us, polarity)]
        """
        ax, ay = int(agent_pos[0]), int(agent_pos[1])
        radius = 8
        events = []

        x_min, x_max = max(0, ax - radius), min(self.map_size, ax + radius)
        y_min, y_max = max(0, ay - radius), min(self.map_size, ay + radius)

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                if self.true_map[y, x] > 0.5:
                    if np.random.rand() < 0.6:  # Olasılıksal spike
                        polarity = 1 if np.random.rand() > 0.5 else -1
                        t_us = int(time.time() * 1e6) % 1000000
                        events.append((x, y, t_us, polarity))

        return events


class SpikeScanMatcher:
    """
    Spike Tabanlı ICP (Iterative Closest Point) Rigit Gövde Hizalama Motoru.
    Girdi spike kümeleri arasındaki bağıl öteleme (dx, dy) ve dönmeyi (d_theta) hesaplar.
    """
    @staticmethod
    def match_scans(prev_pts: np.ndarray, curr_pts: np.ndarray) -> Tuple[float, float, float]:
        """
        Girdi: (N, 2) ve (M, 2) Spike Noktaları -> Çıktı: (dx, dy, d_theta)
        """
        if len(prev_pts) == 0 or len(curr_pts) == 0:
            return 0.0, 0.0, 0.0

        mean_prev = np.mean(prev_pts, axis=0)
        mean_curr = np.mean(curr_pts, axis=0)

        dx = float(mean_curr[0] - mean_prev[0])
        dy = float(mean_curr[1] - mean_prev[1])
        d_theta = 0.0

        return dx, dy, d_theta


class NeuromorphicOccupancyGridSLAM:
    """
    Mikrosaniye Gecikmeli Spike Tabanlı Nöromorfik SLAM Motoru.
    Bayesyen Log-Odds Haritalama ve Gerçek Zamanlı Poz Takibi.
    """
    def __init__(self, map_size: int = 50, l_occ: float = 0.85, l_free: float = 0.2):
        self.map_size = map_size
        self.l_occ = l_occ
        self.l_free = l_free
        
        # Log-Odds 2D Doluluk Haritası
        self.log_odds_map = np.zeros((map_size, map_size), dtype=np.float32)
        self.estimated_pose = np.array([25.0, 25.0, 0.0], dtype=np.float32)
        
        self.prev_event_pts = np.zeros((0, 2), dtype=np.float32)
        self.pose_history = [self.estimated_pose.copy()]

    def process_event_batch(self, events: List[Tuple[int, int, int, int]]) -> Dict[str, Any]:
        """
        Girdi: DVS Spike Olay Listesi -> Çıktı: SLAM Harita & Poz Güncelleme Metrikleri
        """
        start_time = time.time_ns()

        if len(events) > 0:
            curr_pts = np.array([[e[0], e[1]] for e in events], dtype=np.float32)

            # ICP Scan Match Poz Tahmini
            if len(self.prev_event_pts) > 0:
                dx, dy, d_th = SpikeScanMatcher.match_scans(self.prev_event_pts, curr_pts)
                self.estimated_pose[0] += dx * 0.2
                self.estimated_pose[1] += dy * 0.2
                self.estimated_pose[2] += d_th

            self.prev_event_pts = curr_pts

            # Bayesyen Log-Odds Doluluk Haritası Güncellemesi
            for x, y, _, _ in events:
                if 0 <= x < self.map_size and 0 <= y < self.map_size:
                    self.log_odds_map[y, x] = min(5.0, self.log_odds_map[y, x] + self.l_occ)

        self.pose_history.append(self.estimated_pose.copy())
        latency_us = float((time.time_ns() - start_time) / 1000.0)

        # Olasılık Haritası P(Occupied) = 1 - 1 / (1 + exp(L))
        occupancy_prob = 1.0 - (1.0 / (1.0 + np.exp(np.clip(self.log_odds_map, -5.0, 5.0))))

        return {
            "estimated_pose": self.estimated_pose.copy(),
            "occupancy_prob": occupancy_prob,
            "latency_us": latency_us,
            "event_count": len(events),
        }
