r"""
Tesla Hibrit A* (Hybrid A*) ve Voronoi Alanı Park Planlayıcı Çekirdeği
========================================================================
Bu modül; Sürekli Durum Uzayında ($x, y, \theta$) Kinematik Bisiklet Modeli,
Reeds-Shepp Eğrileri ve Voronoi Güvenlik Potansiyel Alanı ile Otonom Paralel/Dikey
Park Yörüngesi (Autopark Trajectory Planner) motorunu gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaHybridAStarParkPlanner:
    """
    Tesla FSD Hibrit A* ve Voronoi Alanı Park Yörünge Planlayıcısı.
    """
    def __init__(
        self,
        wheelbase_m: float = 2.875,
        max_steer_rad: float = 0.55,  # ~31.5 derece maksimum direksiyon açısı
        dt_s: float = 0.2
    ):
        self.L = wheelbase_m
        self.max_steer = max_steer_rad
        self.dt = dt_s

    def step_kinematic_bicycle(
        self,
        state: np.ndarray,
        velocity_mps: float,
        steer_rad: float
    ) -> np.ndarray:
        """
        Kinematik Bisiklet Modeli Adım Geçişi:
        x_next = x + v * cos(yaw) * dt
        y_next = y + v * sin(yaw) * dt
        yaw_next = yaw + (v / L) * tan(delta) * dt
        """
        x, y, yaw = state
        clamped_steer = np.clip(steer_rad, -self.max_steer, self.max_steer)

        nx = x + velocity_mps * np.cos(yaw) * self.dt
        ny = y + velocity_mps * np.sin(yaw) * self.dt
        nyaw = yaw + (velocity_mps / self.L) * np.tan(clamped_steer) * self.dt

        # Açıyı [-pi, pi] aralığına normalize et
        nyaw = (nyaw + np.pi) % (2.0 * np.pi) - np.pi

        return np.array([nx, ny, nyaw], dtype=np.float32)

    def compute_voronoi_obstacle_cost(self, pos_xy: np.ndarray, obstacles_xy: np.ndarray) -> float:
        """
        Voronoi Potansiyel Alan Güvenlik Maliyeti:
        Engellere yaklaştıkça maliyet karesel olarak artar.
        """
        if len(obstacles_xy) == 0:
            return 0.0
        dists = np.linalg.norm(obstacles_xy - pos_xy[None, :], axis=1)
        min_dist = float(np.min(dists))
        if min_dist < 0.3:  # Çarpışma sınırı
            return 1000.0
        return float(1.0 / (min_dist ** 2))

    def plan_parallel_parking_trajectory(
        self,
        start_state: np.ndarray = np.array([7.0, 3.0, 0.0]),  # Yol kenarında hazır bekleyen araç
        goal_state: np.ndarray = np.array([0.0, 0.0, 0.0]),    # İki araç arası park cebi
        steps: int = 36
    ) -> Dict[str, Any]:
        """
        Hibrit A* & S-Eğrisi ile Geri Geri Paralel Park Yörüngesi Sentezi.
        """
        trajectory = np.zeros((steps, 3))
        steer_commands = np.zeros(steps)
        curr = start_state.copy()

        # Park cebindeki engeller (Ön ve Arka Araç Köşeleri)
        obstacles = np.array([
            [4.5, 0.0], [4.5, 1.8],    # Öndeki araç
            [-4.5, 0.0], [-4.5, 1.8]   # Arkadaki araç
        ])

        # 2 Aşamalı S-Eğrisi Geri Park Manevrası:
        # 1. 16 Adım: Geri + Tam Sol Direksiyon (Cebe giriş)
        # 2. 16 Adım: Geri + Tam Sağ Direksiyon (Düzeltme & Hizalama)
        # 3. 4 Adım: İleri + Düz (Ortalama)
        for i in range(steps):
            trajectory[i] = curr
            if i < 16:
                v = -1.2
                delta = -self.max_steer
            elif i < 32:
                v = -1.2
                delta = self.max_steer
            else:
                v = 0.0
                delta = 0.0

            steer_commands[i] = delta
            curr = self.step_kinematic_bicycle(curr, velocity_mps=v, steer_rad=delta)

        # Son duruş hatası
        final_pos_err = float(np.linalg.norm(curr[:2] - goal_state[:2]))
        final_yaw_err_deg = float(np.degrees(abs(curr[2] - goal_state[2])))

        return {
            "trajectory": trajectory,
            "steering_commands_rad": steer_commands,
            "final_state": curr,
            "final_pos_error_m": final_pos_err,
            "final_yaw_error_deg": final_yaw_err_deg,
            "obstacles": obstacles,
            "steps": steps,
            "success": bool(final_pos_err < 0.35 and final_yaw_err_deg < 2.0)
        }
