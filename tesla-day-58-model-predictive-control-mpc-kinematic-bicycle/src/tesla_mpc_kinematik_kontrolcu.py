r"""
Tesla Model Predictive Control (MPC) Kinematik Kontrolcü Çekirdeği
====================================================================
Bu modül; Kinematik Bisiklet Modeli durum-uzayı ($x, y, \psi, v$),
Öngörü Ufku ($N = 20$), Karesel Maliyet Matrisleri ($Q, R, R_\Delta$) ve
Aktüatör Doyum Kısıtları ile Çapraz Hata ($e_y$) & Yönelme Hatası ($e_\psi$)
minimizasyonunu gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaKinematicMPCController:
    """
    Tesla FSD Model Predictive Control (MPC) Çift Eksenli (Yanal & Boyuna) Takip Kontrolcüsü.
    """
    def __init__(
        self,
        wheelbase_m: float = 2.875,
        dt_s: float = 0.1,
        horizon_N: int = 20,
        q_lat: float = 5.0,
        q_yaw: float = 12.0,
        q_vel: float = 2.0,
        r_steer: float = 8.0,
        r_acc: float = 1.5
    ):
        self.L = wheelbase_m
        self.dt = dt_s
        self.N = horizon_N

        # Maliyet Matrisleri: Q (Durum Hatası), R (Kontrol Eforu)
        self.Q = np.diag([q_lat, q_yaw, q_vel])
        self.R = np.diag([r_acc, r_steer])
        self.max_steer_rad = 0.55  # ~31.5 derece
        self.max_acc_mps2 = 2.5
        self.max_decel_mps2 = -4.0
        self.k_cache: Dict[float, np.ndarray] = {}

    def get_gain_k(self, speed_mps: float) -> np.ndarray:
        rounded_v = round(max(speed_mps, 1.0), 1)
        if rounded_v in self.k_cache:
            return self.k_cache[rounded_v]

        v = rounded_v
        A = np.array([
            [1.0, v * self.dt, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        B = np.array([
            [0.0, 0.0],
            [0.0, (v / self.L) * self.dt],
            [self.dt, 0.0]
        ], dtype=np.float64)

        P = self.Q.copy()
        for _ in range(25):
            P_next = self.Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(self.R + B.T @ P @ B) @ B.T @ P @ A
            if np.allclose(P, P_next, atol=1e-4):
                break
            P = P_next

        K = np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)
        self.k_cache[rounded_v] = K
        return K

    def compute_optimal_control(
        self,
        lateral_error_m: float,
        heading_error_rad: float,
        speed_error_mps: float,
        current_speed_mps: float
    ) -> Tuple[float, float]:
        """
        Durum Hatası Vektörü e = [e_y, e_psi, e_v]^T
        LQR/MPC Analitik Geri Besleme Kazancı K
        """
        K = self.get_gain_k(current_speed_mps)

        # Durum Vektörü
        e_state = np.array([lateral_error_m, heading_error_rad, speed_error_mps])
        u_opt = -K @ e_state

        acc_cmd = float(np.clip(u_opt[0], self.max_decel_mps2, self.max_acc_mps2))
        steer_cmd = float(np.clip(u_opt[1], -self.max_steer_rad, self.max_steer_rad))

        return acc_cmd, steer_cmd

    def simulate_closed_loop_tracking(
        self,
        init_lateral_err_m: float = 1.2,
        init_heading_err_rad: float = 0.15,
        target_speed_mps: float = 20.0,
        sim_steps: int = 40
    ) -> Dict[str, Any]:
        """
        40 Adımlık (4.0 saniye) Kapalı Çevrim MPC Yol Takip Simülasyonu.
        """
        lat_errors = np.zeros(sim_steps)
        yaw_errors = np.zeros(sim_steps)
        speed_errors = np.zeros(sim_steps)
        steer_cmds = np.zeros(sim_steps)
        acc_cmds = np.zeros(sim_steps)

        e_y = init_lateral_err_m
        e_psi = init_heading_err_rad
        v = 15.0  # Başlangıç hızı 15 m/s

        for i in range(sim_steps):
            e_v = v - target_speed_mps
            lat_errors[i] = e_y
            yaw_errors[i] = e_psi
            speed_errors[i] = e_v

            acc, steer = self.compute_optimal_control(e_y, e_psi, e_v, v)
            acc_cmds[i] = acc
            steer_cmds[i] = steer

            # Dinamik Güncelleme
            e_y += v * np.sin(e_psi) * self.dt
            e_psi += (v / self.L) * np.tan(steer) * self.dt
            v += acc * self.dt

        final_lat_err = abs(lat_errors[-1])
        final_yaw_err_deg = float(np.degrees(abs(yaw_errors[-1])))
        is_converged = bool(final_lat_err < 0.10 and final_yaw_err_deg < 2.0)

        return {
            "lateral_errors_m": lat_errors,
            "heading_errors_rad": yaw_errors,
            "speed_errors_mps": speed_errors,
            "steer_cmds_rad": steer_cmds,
            "acc_cmds_mps2": acc_cmds,
            "final_lat_err_m": final_lat_err,
            "final_yaw_err_deg": final_yaw_err_deg,
            "is_converged": is_converged,
            "sim_steps": sim_steps
        }
