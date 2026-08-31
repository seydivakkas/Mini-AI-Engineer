r"""
Tesla Frenet Çerçevesi ve Quintic Polinom Şerit Değiştirme Çekirdeği
=====================================================================
Bu modül; Kartezyen (X, Y) <-> Frenet (s, d) Koordinat Dönüşümünü,
5. Derece Quintic Polinom (Jerk-Optimal) Şerit Değiştirme Yörünge Sentezini
ve Sarsıntı (Jerk) / Yanal İvme Konfor Sınırları Analizini gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaFrenetTrajectoryPlanner:
    """
    Frenet Koordinatlarında Jerk-Optimal Quintic Şerit Değiştirme Planlayıcısı.
    """
    def __init__(self, target_lane_width_m: float = 3.5, time_horizon_s: float = 4.0):
        self.lane_width = target_lane_width_m
        self.T = time_horizon_s

    def solve_quintic_polynomial(
        self,
        d0: float, v0: float, a0: float,
        d1: float, v1: float, a1: float,
        T: float
    ) -> np.ndarray:
        """
        d(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
        A * [c3, c4, c5]^T = B
        """
        c0 = d0
        c1 = v0
        c2 = 0.5 * a0

        A = np.array([
            [T**3, T**4, T**5],
            [3.0 * (T**2), 4.0 * (T**3), 5.0 * (T**4)],
            [6.0 * T, 12.0 * (T**2), 20.0 * (T**3)]
        ], dtype=np.float64)

        B = np.array([
            d1 - (d0 + v0 * T + 0.5 * a0 * (T**2)),
            v1 - (v0 + a0 * T),
            a1 - a0
        ], dtype=np.float64)

        c3_4_5 = np.linalg.solve(A, B)
        return np.array([c0, c1, c2, c3_4_5[0], c3_4_5[1], c3_4_5[2]], dtype=np.float64)

    def evaluate_trajectory_profiles(
        self,
        coeffs: np.ndarray,
        time_array: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Konum d(t), Hız d_dot(t), İvme d_ddot(t) ve Jerk d_dddot(t) profilleri.
        """
        c0, c1, c2, c3, c4, c5 = coeffs
        t = time_array

        d = c0 + c1*t + c2*(t**2) + c3*(t**3) + c4*(t**4) + c5*(t**5)
        d_dot = c1 + 2.0*c2*t + 3.0*c3*(t**2) + 4.0*c4*(t**3) + 5.0*c5*(t**4)
        d_ddot = 2.0*c2 + 6.0*c3*t + 12.0*c4*(t**2) + 20.0*c5*(t**3)
        d_dddot = 6.0*c3 + 24.0*c4*t + 60.0*c5*(t**2)

        return {
            "lateral_pos_d": d,
            "lateral_vel_v": d_dot,
            "lateral_acc_a": d_ddot,
            "lateral_jerk_j": d_dddot
        }

    def generate_frenet_lane_change(
        self,
        current_speed_mps: float = 25.0,  # 90 km/h otoyol hızı
        steps: int = 50
    ) -> Dict[str, Any]:
        """
        0'dan 3.5 metreye (komşu şeride) 4 saniyede sarsıntısız geçiş.
        """
        t_arr = np.linspace(0, self.T, steps)
        # Sınır koşulları: d0=0, v0=0, a0=0 -> d1=3.5, v1=0, a1=0
        coeffs = self.solve_quintic_polynomial(d0=0.0, v0=0.0, a0=0.0, d1=self.lane_width, v1=0.0, a1=0.0, T=self.T)
        profiles = self.evaluate_trajectory_profiles(coeffs, t_arr)

        # Boyuna Mesafe s(t) = s0 + v * t
        s_arr = current_speed_mps * t_arr

        # Konfor Analizi: Maksimum Jerk (|jerk| <= 3.5 m/s^3) ve Yanal İvme (|a_lat| <= 2.0 m/s^2)
        max_jerk = float(np.max(np.abs(profiles["lateral_jerk_j"])))
        max_acc = float(np.max(np.abs(profiles["lateral_acc_a"])))
        is_comfortable = bool(max_jerk <= 3.5 and max_acc <= 2.0)

        return {
            "time_array": t_arr,
            "longitudinal_s": s_arr,
            "profiles": profiles,
            "coeffs": coeffs,
            "max_lateral_jerk": max_jerk,
            "max_lateral_acc": max_acc,
            "is_comfortable": is_comfortable,
            "target_lane_width": self.lane_width
        }
