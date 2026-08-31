r"""
Tesla Stanley ve Pure Pursuit Yörünge Takip Kontrolcüsü Çekirdeği
==================================================================
Bu modül; Stanley Kontrolcüsü Ön Aks Geometrik Takip Kanununu,
Pure Pursuit Arka Aks Bakış Noktası (Lookahead) Takipçisini ve
Virajlı Yol Simülasyonunda Çapraz Takip Hatası ($e_{\text{lat}}, e_\psi$)
minimizasyonunu gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaStanleyTracker:
    """
    Tesla FSD Stanley Geometrik Yanal Takip Kontrolcüsü.
    """
    def __init__(
        self,
        gain_k: float = 0.50,
        max_steer_rad: float = 0.55,
        softening_eps: float = 0.10
    ):
        self.k = gain_k
        self.max_steer = max_steer_rad
        self.eps = softening_eps

    def compute_steering(
        self,
        heading_error_rad: float,
        cross_track_error_m: float,
        speed_mps: float
    ) -> float:
        """
        Stanley Kontrol Kanunu:
        delta = theta_e + atan(k * e / (v + eps))
        """
        v = max(speed_mps, 0.0)
        cross_track_term = np.arctan2(self.k * cross_track_error_m, v + self.eps)
        raw_steer = heading_error_rad + cross_track_term
        return float(np.clip(raw_steer, -self.max_steer, self.max_steer))


class TeslaPurePursuitTracker:
    """
    Tesla FSD Pure Pursuit Geometrik Bakış Noktası Takip Kontrolcüsü.
    """
    def __init__(
        self,
        wheelbase_m: float = 2.875,
        lookahead_gain: float = 0.80,
        min_lookahead_m: float = 3.0,
        max_steer_rad: float = 0.55
    ):
        self.L = wheelbase_m
        self.k_look = lookahead_gain
        self.l_min = min_lookahead_m
        self.max_steer = max_steer_rad

    def compute_steering(
        self,
        alpha_rad: float,
        speed_mps: float
    ) -> float:
        """
        Pure Pursuit Kontrol Kanunu:
        delta = atan(2 * L * sin(alpha) / L_d)
        """
        L_d = max(self.k_look * speed_mps, self.l_min)
        raw_steer = np.arctan2(2.0 * self.L * np.sin(alpha_rad), L_d)
        return float(np.clip(raw_steer, -self.max_steer, self.max_steer))


class TeslaTrackingBenchmark:
    """
    Stanley vs Pure Pursuit Karşılaştırmalı Takip Simülatörü.
    """
    def __init__(self):
        self.stanley = TeslaStanleyTracker(gain_k=0.65)
        self.pure_pursuit = TeslaPurePursuitTracker()

    def run_tracking_simulation(self, steps: int = 50, speed_mps: float = 15.0) -> Dict[str, Any]:
        """
        50 Adımlık (2.5s, dt=0.05s) Stanley Kapalı Çevrim Takip Simülasyonu.
        """
        stanley_errors = np.zeros(steps)
        stanley_steers = np.zeros(steps)

        # Başlangıç hatası: 0.30m yanal, 0.04 rad açısal
        e_lat = 0.30
        psi = 0.04
        dt = 0.05
        L = 2.875

        for i in range(steps):
            stanley_errors[i] = e_lat
            theta_e = -psi
            steer = self.stanley.compute_steering(theta_e, e_lat, speed_mps)
            stanley_steers[i] = steer

            # Dinamik kinematik güncelleme
            psi += (speed_mps / L) * np.tan(steer) * dt
            e_lat -= speed_mps * np.sin(psi) * dt

        final_lat_err = float(abs(stanley_errors[-1]))
        is_converged = bool(final_lat_err < 0.05)

        return {
            "stanley_errors_m": stanley_errors,
            "stanley_steers_rad": stanley_steers,
            "final_lateral_error_m": final_lat_err,
            "is_converged": is_converged,
            "steps": steps
        }
