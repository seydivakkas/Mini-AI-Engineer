"""
İnsansı (Humanoid) Robotik Bütünsel Hareket Kontrolü (Whole-Body Control & ZMP) Motoru (Day 251).
3D Doğrusal Ters Sarkaç (LIPM), Destek Poligonu ve Hiyerarşik Denge Optimizasyonu.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.optimize import minimize


class LIPMDynamics:
    """3D Doğrusal Ters Sarkaç Modeli (Linear Inverted Pendulum Model)."""

    def __init__(self, com_yukseklik_m: float = 0.85, yercekimi: float = 9.81):
        self.z_c = com_yukseklik_m
        self.g = yercekimi
        self.omega = np.sqrt(self.g / self.z_c)

    def compute_zmp(self, com_pos: np.ndarray, com_acc: np.ndarray) -> np.ndarray:
        """Sıfır Moment Noktasını (Zero Moment Point - ZMP) hesaplar."""
        # p = x - (z_c / g) * x_ddot
        p_x = com_pos[0] - (self.z_c / self.g) * com_acc[0]
        p_y = com_pos[1] - (self.z_c / self.g) * com_acc[1]
        return np.array([p_x, p_y], dtype=np.float64)


class SupportPolygon:
    """İnsansı Robot Tek/Çift Ayak Destek Poligonu ve Denge Güvenlik Marjini."""

    def __init__(
        self,
        ayak_uzunlugu_m: float = 0.22,
        ayak_genisligi_m: float = 0.12,
        ayak_arasi_mesafe_m: float = 0.20,
    ):
        self.L = ayak_uzunlugu_m
        self.W = ayak_genisligi_m
        self.D = ayak_arasi_mesafe_m

        # Çift ayak destek poligonu sınırları (X: [-0.11, 0.11], Y: [-0.16, 0.16])
        self.x_min = -self.L / 2.0
        self.x_max = self.L / 2.0
        self.y_min = -(self.D / 2.0 + self.W / 2.0)
        self.y_max = (self.D / 2.0 + self.W / 2.0)

    def is_zmp_stable(self, p_zmp: np.ndarray) -> bool:
        """ZMP noktasının destek poligonu içinde olup olmadığını denetler."""
        return bool(
            self.x_min <= p_zmp[0] <= self.x_max
            and self.y_min <= p_zmp[1] <= self.y_max
        )

    def compute_stability_margin(self, p_zmp: np.ndarray) -> float:
        """ZMP'nin en yakın poligon sınırına olan uzaklığı (Pozitif: Güvenli, Negatif: Devriliyor)."""
        dist_x = min(p_zmp[0] - self.x_min, self.x_max - p_zmp[0])
        dist_y = min(p_zmp[1] - self.y_min, self.y_max - p_zmp[1])
        return float(min(dist_x, dist_y))


class HierarchicalQPController:
    """Hiyerarşik Karesel Programlama (QP) Tabanlı Whole-Body Kontrolcüsü."""

    def __init__(self, lipm: LIPMDynamics, polygon: SupportPolygon):
        self.lipm = lipm
        self.polygon = polygon

    def optimize_wbc_step(
        self,
        com_pos: np.ndarray,
        com_vel: np.ndarray,
        hedef_com: np.ndarray,
        dis_kuvvet_N: np.ndarray = np.zeros(2),
        robot_kutlesi_kg: float = 55.0,
    ) -> Dict[str, Any]:
        """Tüm gövde ivme ve torklarını ZMP sınır kısıtı altında optimize eder."""
        # Dış kuvvetten kaynaklanan ivme bozucusu: a_ext = F_ext / m
        a_ext = dis_kuvvet_N / robot_kutlesi_kg

        # Hedef kontrol ivmesi: PD komut a_des = Kp*(x_des - x) - Kd*v
        kp = 15.0
        kd = 6.0
        a_des = kp * (hedef_com[:2] - com_pos[:2]) - kd * com_vel[:2]

        def cost_fn(a_com):
            # Görev 1: Hedef ivmeyi takip et
            tracking_loss = np.sum((a_com - a_des) ** 2)
            # Görev 2: ZMP'yi poligonun merkezine yakın tut
            zmp = self.lipm.compute_zmp(com_pos[:2], a_com + a_ext)
            zmp_center_loss = 0.5 * np.sum(zmp ** 2)
            return float(tracking_loss + zmp_center_loss)

        # Kısıt: ZMP poligon sınırları içinde kalmalıdır
        def constraint_zmp_xmin(a_com):
            zmp = self.lipm.compute_zmp(com_pos[:2], a_com + a_ext)
            return zmp[0] - (self.polygon.x_min + 0.02)

        def constraint_zmp_xmax(a_com):
            zmp = self.lipm.compute_zmp(com_pos[:2], a_com + a_ext)
            return (self.polygon.x_max - 0.02) - zmp[0]

        def constraint_zmp_ymin(a_com):
            zmp = self.lipm.compute_zmp(com_pos[:2], a_com + a_ext)
            return zmp[1] - (self.polygon.y_min + 0.02)

        def constraint_zmp_ymax(a_com):
            zmp = self.lipm.compute_zmp(com_pos[:2], a_com + a_ext)
            return (self.polygon.y_max - 0.02) - zmp[1]

        cons = [
            {"type": "ineq", "fun": constraint_zmp_xmin},
            {"type": "ineq", "fun": constraint_zmp_xmax},
            {"type": "ineq", "fun": constraint_zmp_ymin},
            {"type": "ineq", "fun": constraint_zmp_ymax},
        ]

        res = minimize(
            cost_fn,
            a_des,
            method="SLSQP",
            constraints=cons,
            bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        )

        opt_acc = res.x
        opt_zmp = self.lipm.compute_zmp(com_pos[:2], opt_acc + a_ext)
        marjin = self.polygon.compute_stability_margin(opt_zmp)

        return {
            "opt_com_acc": opt_acc.round(4).tolist(),
            "opt_zmp": opt_zmp.round(4).tolist(),
            "stabil_mi": self.polygon.is_zmp_stable(opt_zmp),
            "guvenlik_marjini_cm": round(marjin * 100.0, 2),
            "optimizasyon_basarili": bool(res.success),
        }
