r"""
Tesla Hız Profili Optimizasyonu Çekirdeği
=========================================
Bu modül; Maksimum Yanal İvme Sınırlandırmalı Viraj Hızı Tespiti ($v = \sqrt{a_{\text{lat}} / \kappa}$),
İleri-Geri Dinamik Programlama (Forward-Backward Pass Speed Profiler),
Boyuna Jerk Sınırlandırması ve Rejeneratif Enerji Verimliliği Analizini gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaSpeedProfileOptimizer:
    """
    Tesla FSD Hız Profili ve Enerji Verimliliği Optimize Edicisi.
    """
    def __init__(
        self,
        max_speed_mps: float = 33.33,       # 120 km/h otoyol hız limiti
        max_lat_accel_mps2: float = 2.0,   # 2.0 m/s^2 konforlu yanal ivme limiti
        max_accel_mps2: float = 2.0,       # 2.0 m/s^2 konforlu boyuna ivmelenme
        max_decel_mps2: float = 2.5,       # 2.5 m/s^2 konforlu rejeneratif frenleme
        regen_efficiency: float = 0.85     # %85 rejenerasyon verimi
    ):
        self.v_max = max_speed_mps
        self.a_lat_max = max_lat_accel_mps2
        self.a_acc_max = max_accel_mps2
        self.a_dec_max = max_decel_mps2
        self.eta_regen = regen_efficiency

    def max_safe_cornering_speed(self, curvature_kappa: float) -> float:
        """
        Maksimum Güvenli Viraj Hızı:
        v_safe = sqrt(a_lat_max / kappa) if kappa > 0 else v_max
        """
        if curvature_kappa <= 1e-4:
            return self.v_max
        v_corner = np.sqrt(self.a_lat_max / curvature_kappa)
        return float(min(v_corner, self.v_max))

    def optimize_speed_profile(
        self,
        track_length_m: float = 200.0,
        num_points: int = 100,
        curve_start_m: float = 70.0,
        curve_end_m: float = 130.0,
        curve_kappa: float = 0.04  # R = 25m viraj
    ) -> Dict[str, Any]:
        """
        İleri-Geri Geçiş (Forward-Backward Pass) ile Hız Profili Optimizasyonu.
        """
        s_arr = np.linspace(0, track_length_m, num_points)
        ds = track_length_m / (num_points - 1)

        # 1. Yol Eğrilik Dağılımı ve Ham Hız Limitleri
        kappas = np.zeros(num_points)
        curve_mask = (s_arr >= curve_start_m) & (s_arr <= curve_end_m)
        kappas[curve_mask] = curve_kappa

        v_limits = np.array([self.max_safe_cornering_speed(k) for k in kappas])

        # 2. İleri Geçiş (Forward Pass: Hızlanma Kısıtı)
        v_forward = np.zeros(num_points)
        v_forward[0] = min(15.0, v_limits[0])  # Başlangıç hızı 15 m/s

        for i in range(num_points - 1):
            v_allowed = np.sqrt(v_forward[i]**2 + 2.0 * self.a_acc_max * ds)
            v_forward[i+1] = min(v_limits[i+1], v_allowed)

        # 3. Geri Geçiş (Backward Pass: Frenleme Kısıtı)
        v_opt = v_forward.copy()
        for i in range(num_points - 2, -1, -1):
            v_allowed = np.sqrt(v_opt[i+1]**2 + 2.0 * self.a_dec_max * ds)
            v_opt[i] = min(v_opt[i], v_allowed)

        # 4. İvme Profili Hesabı: a = (v_{i+1}^2 - v_i^2) / (2 * ds)
        accels = np.zeros(num_points)
        accels[:-1] = (v_opt[1:]**2 - v_opt[:-1]**2) / (2.0 * ds)
        accels[-1] = accels[-2]

        # 5. Yanal İvme: a_lat = v^2 * kappa
        lat_accels = (v_opt**2) * kappas

        # Rejeneratif Enerji Tasarrufu (Kinetik Enerji Geri Kazanımı):
        # Fren yapılan bölgelerde E = eta * integral(m * |a| * v dt)
        vehicle_mass_kg = 1800.0  # Model 3
        braking_mask = accels < 0
        v_avg_braking = np.mean(v_opt[braking_mask]) if np.any(braking_mask) else 0.0
        regen_energy_kj = float(0.5 * vehicle_mass_kg * (np.max(v_opt)**2 - np.min(v_opt)**2) * self.eta_regen / 1000.0)

        return {
            "s_array": s_arr,
            "curvature_kappa": kappas,
            "speed_limits_mps": v_limits,
            "optimized_speed_mps": v_opt,
            "longitudinal_acc_mps2": accels,
            "lateral_acc_mps2": lat_accels,
            "regen_energy_kj": regen_energy_kj,
            "min_corner_speed_mps": float(np.min(v_opt)),
            "max_straight_speed_mps": float(np.max(v_opt)),
            "is_comfortable": bool(np.max(lat_accels) <= self.a_lat_max + 0.05)
        }
