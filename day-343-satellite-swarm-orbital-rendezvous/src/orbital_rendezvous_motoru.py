"""
Day 343: Satellite Swarm Orbital Rendezvous & Autonomous Collision Avoidance
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Hill-Clohessy-Wiltshire (HCW/CW) Bağıl Yörünge Hareket Modellerini,
Yapay Potansiyel Alanı (APF) Çarpışma Önleme ve Otonom Buluşma/Kenetlenme Kontrolcüsünü içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np


class ClohessyWiltshirePropagator:
    """
    Hill-Clohessy-Wiltshire (HCW/CW) Bağıl Yörünge Hareketi Dinamiği.
    Ana Uydu (Chief) merkezli LVLH (Local Vertical, Local Horizontal) koordinat çerçevesinde
    yardımcı uyduların (Deputy Swarm) bağıl pozisyon [x, y, z] ve hız [vx, vy, vz] vektörlerini öteler.
    """
    def __init__(self, orbital_radius_km: float = 7000.0, mu: float = 398600.4418):
        self.r_orbit = orbital_radius_km
        self.mu = mu
        # Ortalama yörünge hareketi n (rad/s)
        self.n = math.sqrt(self.mu / (self.r_orbit ** 3))

    def step(self, state: np.ndarray, u_thrust: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        CW diferansiyel denklemlerini entegre eder:
          x_ddot =  2*n*y_dot + 3*n^2*x + u_x
          y_ddot = -2*n*x_dot          + u_y
          z_ddot = -n^2*z              + u_z
        """
        x, y, z, vx, vy, vz = state
        ux, uy, uz = u_thrust

        ax = 2.0 * self.n * vy + 3.0 * (self.n ** 2) * x + ux
        ay = -2.0 * self.n * vx + uy
        az = - (self.n ** 2) * z + uz

        new_x = x + vx * dt + 0.5 * ax * (dt ** 2)
        new_y = y + vy * dt + 0.5 * ay * (dt ** 2)
        new_z = z + vz * dt + 0.5 * az * (dt ** 2)

        new_vx = vx + ax * dt
        new_vy = vy + ay * dt
        new_vz = vz + az * dt

        return np.array([new_x, new_y, new_z, new_vx, new_vy, new_vz])


class SwarmPotentialFieldCollisionAvoidance:
    """
    Yapay Potansiyel Alanı (Artificial Potential Field - APF) Çarpışma Önleme.
    Sürü uyduları arasında güvenli yaklaşma mesafesi (d0_m) altına inildiğinde itici kuvvet (F_rep) üretir.
    """
    def __init__(self, d_safe_m: float = 20.0, k_rep: float = 0.5):
        self.d_safe_m = d_safe_m
        self.k_rep = k_rep

    def compute_repulsion(self, current_pos: np.ndarray, other_positions: List[np.ndarray]) -> np.ndarray:
        """
        Diğer sürü uydularından gelen itici kaçınma kuvvetini hesaplar (km/s^2).
        """
        f_rep_m_s2 = np.zeros(3)
        pos_m = current_pos * 1000.0

        for other_p in other_positions:
            other_m = other_p * 1000.0
            diff_m = pos_m - other_m
            dist_m = float(np.linalg.norm(diff_m))
            
            if 0.1 < dist_m < self.d_safe_m:
                # İtici kuvvet (m/s^2)
                mag = self.k_rep * (1.0 / dist_m - 1.0 / self.d_safe_m)
                direction = diff_m / dist_m
                f_rep_m_s2 += mag * direction

        return f_rep_m_s2 / 1000.0 # km/s^2 cinsine dönüştür


class AutonomousRendezvousController:
    """
    Otonom Uydu Sürüsü Buluşma (Rendezvous & Docking) Kontrolcüsü.
    Hedef kenetlenme limanına doğru çekici kuvvet ile sürü içi çarpışma kaçınma kuvvetini birleştirir.
    """
    def __init__(self, propagator: ClohessyWiltshirePropagator, apf: SwarmPotentialFieldCollisionAvoidance, kp: float = 0.001, kd: float = 0.02):
        self.prop = propagator
        self.apf = apf
        self.kp = kp
        self.kd = kd

    def compute_docking_control(
        self,
        deputy_state: np.ndarray,
        target_docking_pos: np.ndarray,
        other_deputy_positions: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Hedefe yaklaşma ve sürüden kaçınma itki vektörünü hesaplar (CW Geri Besleme Doğrusallaştırması).
        """
        pos = deputy_state[:3]
        vel = deputy_state[3:]
        n = self.prop.n

        # CW dinamik terimleri telafi ivmesi (Feedback Linearization)
        a_comp = np.array([
            - 2.0 * n * vel[1] - 3.0 * (n ** 2) * pos[0],
              2.0 * n * vel[0],
              (n ** 2) * pos[2]
        ])

        # Hedefe çekim kuvveti (Kararlı Ayrık PD)
        pos_err = target_docking_pos - pos
        f_attr = 0.05 * pos_err - 0.3 * vel + a_comp

        # Sürü içi çarpışma kaçınma itici kuvveti
        f_rep = self.apf.compute_repulsion(pos, other_deputy_positions)

        total_thrust = f_attr + f_rep
        thrust_mag = float(np.linalg.norm(total_thrust) * 1000.0) # m/s^2

        return {
            "u_thrust": total_thrust,
            "thrust_magnitude_m_s2": thrust_mag,
            "dist_to_docking_m": float(np.linalg.norm(pos_err) * 1000.0),
            "is_docked": float(np.linalg.norm(pos_err) * 1000.0) < 0.5
        }
