"""
Day 347: Decentralized Drone Swarm Flocking with Graph Neural Networks (GNN)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Dinamik İHA Komşuluk Grafını, Graf Sinir Ağı (GNN Message Passing) Tabanlı
Merkeziyetsiz Sürü (Flocking) Kontrolcüsünü ve 3D Sürü Uçuş Simülatörünü içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np


class DroneSwarmDynamicGraph:
    """
    Dinamik İHA Sürüsü Graf Yapısı G(t) = (V, E(t)).
    İletişim menzili (R_comm) içindeki İHA'lar arasında kenarlar oluşturur.
    """
    def __init__(self, num_drones: int = 15, comm_radius_m: float = 60.0):
        self.N = num_drones
        self.r_comm = comm_radius_m

    def build_adjacency(self, positions: np.ndarray) -> np.ndarray:
        """
        İHA pozisyonlarından ikili komşuluk matrisi A (NxN) ve kenar listesi oluşturur.
        """
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :] # (N, N, 3)
        dists = np.linalg.norm(diff, axis=-1) # (N, N)
        
        adj = (dists <= self.r_comm) & (dists > 1e-4) # Kendisi hariç komşular
        return adj.astype(float)


class GNNFlockingController:
    """
    Graf Sinir Ağları (GNN) Tabanlı Merkeziyetsiz Sürü Davranışı (Flocking) Kontrolcüsü.
    Düğüm Özellikleri (Node Features) ve Kenar Özellikleri (Edge Features) üzerinden
    Mesaj Geçirme (Message Passing) ile her İHA için ivme komutu u_i üretir.
    """
    def __init__(self, d_desired_m: float = 20.0, k_sep: float = 1.2, k_align: float = 0.8, k_coh: float = 0.5):
        self.d_des = d_desired_m
        self.k_sep = k_sep
        self.k_align = k_align
        self.k_coh = k_coh

    def compute_gnn_control(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        adj_matrix: np.ndarray,
        target_waypoint: np.ndarray
    ) -> np.ndarray:
        """
        Her İHA için GNN mesaj geçirme mantığıyla ivme komutlarını hesaplar (N, 3).
        """
        N = len(positions)
        u_commands = np.zeros((N, 3))

        for i in range(N):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            
            f_sep = np.zeros(3)
            f_align = np.zeros(3)
            f_coh = np.zeros(3)

            if len(neighbors) > 0:
                # 1. Ayrılma (Separation): Çok yakın komşulardan kaçınma
                for j in neighbors:
                    diff_p = positions[i] - positions[j]
                    dist = float(np.linalg.norm(diff_p))
                    if dist < self.d_des:
                        f_sep += (diff_p / (dist + 1e-3)) * (self.d_des - dist)

                # 2. Hizalanma (Alignment): Komşularla hız vektörü mutabakatı (Consensus)
                mean_neighbor_vel = np.mean(velocities[neighbors], axis=0)
                f_align = mean_neighbor_vel - velocities[i]

                # 3. Bütünleşme (Cohesion): Komşuların geometrik merkezine yönelme
                mean_neighbor_pos = np.mean(positions[neighbors], axis=0)
                f_coh = mean_neighbor_pos - positions[i]

            # 4. Görev Hedefine Çekim (Navigasyon Vektörü)
            f_goal = (target_waypoint - positions[i])
            if np.linalg.norm(f_goal) > 0:
                f_goal = (f_goal / np.linalg.norm(f_goal)) * 5.0

            u_i = self.k_sep * f_sep + self.k_align * f_align + self.k_coh * f_coh + 0.3 * f_goal
            # Maksimum ivme doyumu (Max acceleration clipping: 10 m/s^2)
            if np.linalg.norm(u_i) > 10.0:
                u_i = (u_i / np.linalg.norm(u_i)) * 10.0

            u_commands[i] = u_i

        return u_commands


class DecentralizedSwarmSimulator:
    """
    3D İHA Sürüsü Merkeziyetsiz Uçuş Simülatörü.
    """
    def __init__(self, num_drones: int = 15, dt: float = 0.05):
        self.num_drones = num_drones
        self.dt = dt
        self.graph_builder = DroneSwarmDynamicGraph(num_drones=num_drones)
        self.controller = GNNFlockingController()

        # Başlangıç Rastgele Pozisyonlar ve Hızlar
        np.random.seed(42)
        self.positions = np.random.uniform(-30, 30, (num_drones, 3))
        self.positions[:, 2] = np.random.uniform(50, 70, num_drones) # İrtifa (m)
        self.velocities = np.random.uniform(-2, 2, (num_drones, 3))

    def step_simulation(self, target_waypoint: np.ndarray) -> Dict[str, Any]:
        """Sürü fiziğini bir adım ilerletir."""
        adj = self.graph_builder.build_adjacency(self.positions)
        u_acc = self.controller.compute_gnn_control(self.positions, self.velocities, adj, target_waypoint)

        self.velocities += u_acc * self.dt
        # Hız sınırı (Max speed: 20 m/s)
        speeds = np.linalg.norm(self.velocities, axis=-1, keepdims=True)
        speeds_clipped = np.clip(speeds, 0.0, 20.0)
        self.velocities = np.where(speeds > 1e-4, (self.velocities / (speeds + 1e-6)) * speeds_clipped, self.velocities)

        self.positions += self.velocities * self.dt

        # Sürü İçi Metrikler
        dists = []
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                dists.append(float(np.linalg.norm(self.positions[i] - self.positions[j])))

        min_dist = float(np.min(dists)) if len(dists) > 0 else 999.0
        
        # Hız Hizalanma Varyansı (Alignment Consensus)
        vel_variance = float(np.mean(np.var(self.velocities, axis=0)))

        return {
            "positions": self.positions.copy(),
            "velocities": self.velocities.copy(),
            "min_distance_m": min_dist,
            "velocity_variance": vel_variance,
            "has_collision": min_dist < 2.0 # 2 metre altı çarpışma sayılır
        }
