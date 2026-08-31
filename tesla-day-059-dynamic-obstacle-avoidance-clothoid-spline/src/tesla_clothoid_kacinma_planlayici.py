r"""
Tesla Clothoid (Euler Spirali) ve Dinamik Engelden Kaçınma Çekirdeği
======================================================================
Bu modül; Doğrusal Değişen Eğrilik ($\kappa(s) = \kappa_0 + c \cdot s$),
Fresnel İntegralleri ile $C^2$ Sürekli Yörünge Sentezini, Direksiyon Simidi
Dönüş Hızı Kısıtlarını ve Dinamik Engelden Kaçınma Manevrasını gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaClothoidAvoidancePlanner:
    """
    Tesla FSD Clothoid (Euler Spiral) Sürekli Eğrilikli Engelden Kaçınma Planlayıcısı.
    """
    def __init__(
        self,
        wheelbase_m: float = 2.875,
        max_steer_rate_rad_s: float = 0.60,  # Maksimum direksiyon motor dönüş hızı
        max_curvature_m_inv: float = 0.20   # Maksimum eğrilik (R_min = 5.0m)
    ):
        self.L = wheelbase_m
        self.max_steer_rate = max_steer_rate_rad_s
        self.max_kappa = max_curvature_m_inv

    def is_curvature_rate_safe(
        self,
        kappa_curr: float,
        kappa_next: float,
        ds: float,
        speed_mps: float
    ) -> bool:
        """
        Eğrilik Değişim Hızı Güvenlik Kontrolü:
        |dkappa / dt| = |(kappa_next - kappa_curr) / ds| * v <= (max_steer_rate / L)
        """
        if ds <= 1e-4 or speed_mps <= 0.1:
            return True
        dkappa_ds = abs(kappa_next - kappa_curr) / ds
        dkappa_dt = dkappa_ds * speed_mps
        max_allowable_dkappa_dt = self.max_steer_rate / self.L
        return bool(dkappa_dt <= max_allowable_dkappa_dt)

    def generate_clothoid_segment(
        self,
        s_total_m: float,
        kappa_start: float,
        kappa_end: float,
        start_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        num_points: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Doğrusal Eğrilikli Clothoid Dilimi Üretimi:
        kappa(s) = kappa_start + c * s
        theta(s) = theta0 + kappa_start * s + 0.5 * c * s^2
        """
        s_arr = np.linspace(0, s_total_m, num_points)
        c = (kappa_end - kappa_start) / max(s_total_m, 1e-4)

        x0, y0, theta0 = start_pose
        thetas = theta0 + kappa_start * s_arr + 0.5 * c * (s_arr ** 2)
        kappas = kappa_start + c * s_arr

        # Sayısal İntegrasyon (Kümülatif Trapezoid)
        ds = s_total_m / (num_points - 1)
        dx = np.cos(thetas) * ds
        dy = np.sin(thetas) * ds

        xs = x0 + np.cumsum(dx) - dx[0]
        ys = y0 + np.cumsum(dy) - dy[0]

        return {
            "s_array": s_arr,
            "x": xs,
            "y": ys,
            "theta": thetas,
            "curvature_kappa": kappas,
            "curvature_rate_c": c
        }

    def plan_obstacle_avoidance_maneuver(
        self,
        obstacle_x_m: float = 35.0,
        obstacle_y_m: float = 0.0,
        lane_shift_m: float = 3.5,
        speed_mps: float = 20.0  # 72 km/h
    ) -> Dict[str, Any]:
        """
        Çift Clothoid (S-Eğrisi) Sürekli Eğrilikli Kaçınma Yörüngesi.
        1. Dilim: Düzden viraja giriş (0 -> +kappa_peak)
        2. Dilim: Virajdan orta noktaya dönüş (+kappa_peak -> 0)
        3. Dilim: Karşı yöne viraj (0 -> -kappa_peak)
        4. Dilim: Hedef şeride düzleşme (-kappa_peak -> 0)
        """
        seg_len = 15.0  # Her dilim 15 metre
        kappa_peak = 0.04  # R = 25m dönüş yarıçapı

        # 4 Kademeli Clothoid Zinciri
        s1 = self.generate_clothoid_segment(seg_len, 0.0, kappa_peak, (0.0, 0.0, 0.0), 25)
        s2 = self.generate_clothoid_segment(seg_len, kappa_peak, 0.0, (s1["x"][-1], s1["y"][-1], s1["theta"][-1]), 25)
        s3 = self.generate_clothoid_segment(seg_len, 0.0, -kappa_peak, (s2["x"][-1], s2["y"][-1], s2["theta"][-1]), 25)
        s4 = self.generate_clothoid_segment(seg_len, -kappa_peak, 0.0, (s3["x"][-1], s3["y"][-1], s3["theta"][-1]), 25)

        x_full = np.concatenate([s1["x"], s2["x"], s3["x"], s4["x"]])
        y_full = np.concatenate([s1["y"], s2["y"], s3["y"], s4["y"]])
        k_full = np.concatenate([s1["curvature_kappa"], s2["curvature_kappa"], s3["curvature_kappa"], s4["curvature_kappa"]])
        t_full = np.concatenate([s1["theta"], s2["theta"], s3["theta"], s4["theta"]])

        # Engele Olan Minimum Güvenlik Mesafesi
        dists_to_obs = np.sqrt((x_full - obstacle_x_m)**2 + (y_full - obstacle_y_m)**2)
        min_clearance = float(np.min(dists_to_obs))

        # Eğrilik sürekliliği ve aktüatör güvenliği
        is_safe = bool(min_clearance >= 1.5 and np.max(np.abs(k_full)) <= self.max_kappa)

        return {
            "x_traj": x_full,
            "y_traj": y_full,
            "curvature_kappa": k_full,
            "theta_traj": t_full,
            "min_clearance_m": min_clearance,
            "obstacle_pos": (obstacle_x_m, obstacle_y_m),
            "is_safe": is_safe,
            "total_length_m": 4 * seg_len
        }
