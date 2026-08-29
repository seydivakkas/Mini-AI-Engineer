"""
Model Predictive Control (MPC) ile Yüksek Hızlı Dinamik Engelden Kaçınma Motoru (Day 257).
Kayan Ufuk (Receding Horizon N=15), Dinamik Engel Hız Projeksiyonu ve Güvenlik Bariyeri Optimizasyonu.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.optimize import minimize


class DynamicObstacleTracker:
    """Hareketli Engellerin Hız Vektörlerini ve Gelecek Konumlarını Takip Eden Modül."""

    def __init__(
        self,
        pos_init: Tuple[float, float],
        vel_xy: Tuple[float, float],
        radius: float = 0.35,
    ):
        self.pos = np.array(pos_init, dtype=np.float64)
        self.vel = np.array(vel_xy, dtype=np.float64)
        self.radius = radius

    def predict_trajectory(self, horizon: int = 15, dt: float = 0.1) -> np.ndarray:
        """Gelecek N adımlık konum dizisini [N, 2] olarak projekte eder."""
        traj = np.zeros((horizon, 2), dtype=np.float64)
        for k in range(horizon):
            traj[k] = self.pos + self.vel * (k * dt)
        return traj

    def update_state(self, dt: float = 0.1):
        """Engelin gerçek zamanlı konumunu günceller."""
        self.pos += self.vel * dt


class NonlinearMPCController:
    """Kayan Ufuklu Dinamik Engelden Kaçınma NMPC Kontrolcüsü."""

    def __init__(
        self,
        horizon: int = 15,
        dt: float = 0.1,
        robot_radius: float = 0.30,
        v_max: float = 3.0,
    ):
        self.N = horizon
        self.dt = dt
        self.robot_radius = robot_radius
        self.v_max = v_max

        # Durum: [x, y, theta, v]
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def solve_control(
        self,
        current_state: np.ndarray,
        goal_pos: Tuple[float, float],
        dynamic_obstacles: List[DynamicObstacleTracker],
        v_target: float = 2.4,
    ) -> Dict[str, Any]:
        """Kayan ufuk optimizasyonu ile en uygun ivme ve açısal hız aksiyonunu çözer."""
        self.state = np.array(current_state, dtype=np.float64)
        goal = np.array(goal_pos, dtype=np.float64)

        # Engellerin N adımlık gelecek tahminleri
        obs_trajs = [obs.predict_trajectory(self.N, self.dt) for obs in dynamic_obstacles]
        obs_radii = [obs.radius for obs in dynamic_obstacles]

        # Karar değişkenleri: u = [a_0, omega_0, a_1, omega_1, ..., a_{N-1}, omega_{N-1}] (2N)
        u_init = np.zeros(2 * self.N)
        # İvme sınırları [-2.0, 2.0] m/s^2, Açısal Hız sınırları [-1.5, 1.5] rad/s
        bounds = []
        for _ in range(self.N):
            bounds.append((-2.0, 2.0))
            bounds.append((-1.5, 1.5))

        def cost_func(u_flat):
            u_mat = u_flat.reshape(self.N, 2)
            total_cost = 0.0
            x, y, theta, v = self.state

            for k in range(self.N):
                a_k, omega_k = u_mat[k]

                # Kinematik Model Güncellemesi
                x += v * np.cos(theta) * self.dt
                y += v * np.sin(theta) * self.dt
                theta += omega_k * self.dt
                v = np.clip(v + a_k * self.dt, 0.0, self.v_max)

                # 1. Hedefe Yaklaşma Maliyeti
                dist_to_goal = np.linalg.norm(np.array([x, y]) - goal)
                total_cost += 1.5 * (dist_to_goal ** 2)

                # 2. Hedef Hız Takip Maliyeti
                total_cost += 0.8 * ((v - v_target) ** 2)

                # 3. Kontrol Eforu Cezası
                total_cost += 0.05 * (a_k ** 2 + omega_k ** 2)

                # 4. Dinamik Engel Güvenlik Bariyeri Potansiyeli
                for obs_idx, obs_tr in enumerate(obs_trajs):
                    obs_pos_k = obs_tr[k]
                    d_safe = self.robot_radius + obs_radii[obs_idx] + 0.25
                    dist_to_obs = np.linalg.norm(np.array([x, y]) - obs_pos_k)

                    if dist_to_obs < d_safe:
                        total_cost += 200.0 * ((d_safe - dist_to_obs) ** 2)

            return total_cost

        res = minimize(cost_func, u_init, method="SLSQP", bounds=bounds, options={"maxiter": 30})

        opt_u = res.x.reshape(self.N, 2)
        best_accel, best_omega = opt_u[0]

        # Öngörülen Gelecek Robot Yörüngesi
        pred_traj = np.zeros((self.N, 4))
        x, y, theta, v = self.state
        for k in range(self.N):
            a_k, omega_k = opt_u[k]
            x += v * np.cos(theta) * self.dt
            y += v * np.sin(theta) * self.dt
            theta += omega_k * self.dt
            v = np.clip(v + a_k * self.dt, 0.0, self.v_max)
            pred_traj[k] = [x, y, theta, v]

        return {
            "en_iyi_ivme_m_s2": round(float(best_accel), 3),
            "en_iyi_acisal_hiz_rad_s": round(float(best_omega), 3),
            "ongorulen_robot_yorungesi": pred_traj,
            "engel_gelecek_yorungeleri": obs_trajs,
            "optimizasyon_basarili_mi": bool(res.success),
        }
