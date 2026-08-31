r"""
Tesla Optimus Bütünsel Denge (Whole-Body Locomotion) ve ZMP Kontrol Çekirdeği
=============================================================================
Bu modül; Tesla Optimus iki ayaklı insansı robotunun Sıfır An Moment Noktası
(Zero Moment Point - ZMP), Doğrusal Ters Sarkaç Modeli (LIPM), Destek Poligonu
kararlılık analizi, Denge Kurtarma (Push Recovery) ve Yakalama Noktası
(Capture Point) adımlama algoritmasını gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaOptimusZMPBalanceController:
    """
    Tesla Optimus ZMP ve Bütünsel Denge Kontrolcüsü.
    """
    def __init__(
        self,
        robot_mass_kg: float = 56.0,
        nominal_com_height_m: float = 0.85,
        foot_length_m: float = 0.27,
        foot_width_m: float = 0.12,
        stance_width_m: float = 0.28
    ):
        self.mass = robot_mass_kg
        self.z_com = nominal_com_height_m
        self.g = 9.81
        self.omega_0 = np.sqrt(self.g / self.z_com)  # Doğal LIPM frekansı (rad/s)

        # Çift Ayak Destek Poligonu Sınırları (m)
        self.x_poly_min = -foot_length_m * 0.4
        self.x_poly_max = foot_length_m * 0.6
        self.y_poly_min = -stance_width_m / 2.0 - foot_width_m / 2.0
        self.y_poly_max = stance_width_m / 2.0 + foot_width_m / 2.0

    def compute_zmp(
        self,
        x_com: float,
        y_com: float,
        x_ddot_com: float,
        y_ddot_com: float,
        z_com: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        LIPM Sıfır An Moment Noktası (ZMP) Koordinatlarını Hesaplar:
        x_zmp = x_com - (z_com / g) * x_ddot_com
        y_zmp = y_com - (z_com / g) * y_ddot_com
        """
        z_h = z_com if z_com is not None else self.z_com
        x_zmp = x_com - (z_h / self.g) * x_ddot_com
        y_zmp = y_com - (z_h / self.g) * y_ddot_com
        return float(x_zmp), float(y_zmp)

    def is_zmp_within_support(self, x_zmp: float, y_zmp: float) -> bool:
        """ZMP'nin ayak destek poligonu içinde olup olmadığını kontrol eder."""
        x_ok = (self.x_poly_min <= x_zmp <= self.x_poly_max)
        y_ok = (self.y_poly_min <= y_zmp <= self.y_poly_max)
        return bool(x_ok and y_ok)

    def compute_capture_point(
        self,
        x_com: float,
        y_com: float,
        x_dot_com: float,
        y_dot_com: float
    ) -> Tuple[float, float]:
        """
        Dengeyi sağlamak için ayağın basması gereken Capture Point (Yakalama Noktası):
        x_cp = x_com + x_dot_com / omega_0
        y_cp = y_com + y_dot_com / omega_0
        """
        x_cp = x_com + (x_dot_com / self.omega_0)
        y_cp = y_com + (y_dot_com / self.omega_0)
        return float(x_cp), float(y_cp)

    def push_recovery_step(
        self,
        x_com: float,
        y_com: float,
        x_dot_com: float,
        y_dot_com: float,
        ext_impulse_ns: float = 25.0
    ) -> Dict[str, Any]:
        """
        Robot dışarıdan bir itme (Push / Impulse) aldığında denge kurtarma stratejisini belirler.
        """
        # İtme sonucu oluşan anlık hız değişimi: delta_v = Impulse / Mass
        delta_vx = ext_impulse_ns / self.mass
        x_dot_new = x_dot_com + delta_vx

        x_cp, y_cp = self.compute_capture_point(x_com, y_com, x_dot_new, y_dot_com)
        x_zmp, y_zmp = self.compute_zmp(x_com, y_com, x_ddot_com=delta_vx * 10.0, y_ddot_com=0.0)

        # Destek poligonunda mı?
        zmp_safe = self.is_zmp_within_support(x_zmp, y_zmp)
        cp_safe = (self.x_poly_min <= x_cp <= self.x_poly_max) and (self.y_poly_min <= y_cp <= self.y_poly_max)

        if cp_safe and zmp_safe:
            strategy = "ANKLE_STRATEGY (Bilek Torku Yeterli)"
            step_required = False
        elif not cp_safe and ext_impulse_ns > 40.0:
            strategy = "STEPPING_STRATEGY (Adım Atma Zorunlu)"
            step_required = True
        else:
            strategy = "HIP_STRATEGY (Kalça Fleksiyonu Dengeleme)"
            step_required = False

        return {
            "impulse_ns": ext_impulse_ns,
            "x_zmp_m": round(x_zmp, 4),
            "y_zmp_m": round(y_zmp, 4),
            "x_cp_m": round(x_cp, 4),
            "y_cp_m": round(y_cp, 4),
            "zmp_safe": zmp_safe,
            "recovery_strategy": strategy,
            "step_required": step_required
        }
