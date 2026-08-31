"""
Tesla PMSM İnvertör ve Field Oriented Control (FOC) Kontrolcüsü
================================================================
Bu modül; Tesla Model 3 IPM-SynRM (Sabit Mıknatıslı Senkron Relüktans)
çekiş motorunun Clarke, Park, Ters Park dönüşümlerini ve dq-ekseni
ayrıştırılmış akım/tork kontrol döngülerini (FOC) gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


class ClarkeTransform:
    """
    3-Faz Sabit Stator Çerçevesinden 2-Faz Sabit Çerçeveye Dönüşüm (abc -> alpha-beta).
    Genlik Değişmez (Amplitude Invariant) Biçim:
    i_alpha = i_a
    i_beta  = (1 / sqrt(3)) * (i_a + 2*i_b)
    """
    @staticmethod
    def forward(i_a: float, i_b: float, i_c: float) -> Tuple[float, float]:
        i_alpha = float(i_a)
        i_beta = float((1.0 / np.sqrt(3.0)) * (i_a + 2.0 * i_b))
        return i_alpha, i_beta

    @staticmethod
    def inverse(v_alpha: float, v_beta: float) -> Tuple[float, float, float]:
        v_a = float(v_alpha)
        v_b = float(-0.5 * v_alpha + (np.sqrt(3.0) / 2.0) * v_beta)
        v_c = float(-0.5 * v_alpha - (np.sqrt(3.0) / 2.0) * v_beta)
        return v_a, v_b, v_c


class ParkTransform:
    """
    2-Faz Sabit Çerçeveden Dönen Rotor Çerçevesine Dönüşüm (alpha-beta -> dq).
    i_d =  i_alpha * cos(theta_e) + i_beta * sin(theta_e)
    i_q = -i_alpha * sin(theta_e) + i_beta * cos(theta_e)
    """
    @staticmethod
    def forward(i_alpha: float, i_beta: float, theta_e_rad: float) -> Tuple[float, float]:
        cos_th = np.cos(theta_e_rad)
        sin_th = np.sin(theta_e_rad)
        i_d = float(i_alpha * cos_th + i_beta * sin_th)
        i_q = float(-i_alpha * sin_th + i_beta * cos_th)
        return i_d, i_q

    @staticmethod
    def inverse(v_d: float, v_q: float, theta_e_rad: float) -> Tuple[float, float]:
        cos_th = np.cos(theta_e_rad)
        sin_th = np.sin(theta_e_rad)
        v_alpha = float(v_d * cos_th - v_q * sin_th)
        v_beta = float(v_d * sin_th + v_q * cos_th)
        return v_alpha, v_beta


class PIController:
    """
    Anti-Windup Korumalı Ayrık Zamanlı PI Akım Kontrolörü.
    """
    def __init__(self, kp: float, ki: float, output_limit: float = 350.0):
        self.kp = kp
        self.ki = ki
        self.limit = output_limit
        self.integral = 0.0

    def step(self, target: float, actual: float, dt_s: float = 0.0001) -> float:
        error = target - actual
        self.integral += error * self.ki * dt_s
        # Anti-windup clamping
        self.integral = float(np.clip(self.integral, -self.limit, self.limit))
        output = self.kp * error + self.integral
        return float(np.clip(output, -self.limit, self.limit))


@dataclass
class TeslaMotorParameters:
    pole_pairs: int = 4              # 8 Kutuplu PMSM Motor
    psi_f_wb: float = 0.175          # Sabit Mıknatıs Akısı (Wb)
    l_d_h: float = 0.00035           # d-Ekseni Endüktansı (350 µH)
    l_q_h: float = 0.00075           # q-Ekseni Endüktansı (750 µH - Relüktans Torku için Lq > Ld)
    r_s_ohm: float = 0.015           # Faz Direnci (15 mΩ)
    max_current_a: float = 450.0     # Maksimum Faz Akımı (Pik)
    v_dc_bus: float = 400.0          # Batarya DC Bara Gerilimi (V)


class TeslaFOCController:
    """
    Tesla FOC (Field Oriented Control) 10 kHz Motor Kontrol Çekirdeği.
    """
    def __init__(self, motor_params: TeslaMotorParameters):
        self.params = motor_params
        # Akım döngüsü PI kazançları
        self.pi_d = PIController(kp=1.2, ki=450.0, output_limit=motor_params.v_dc_bus / np.sqrt(3.0))
        self.pi_q = PIController(kp=1.5, ki=550.0, output_limit=motor_params.v_dc_bus / np.sqrt(3.0))

    def compute_electromagnetic_torque(self, i_d: float, i_q: float) -> float:
        """
        Elektromanyetik Tork Hesabı (Manyetik Tork + Relüktans Torku):
        Te = 1.5 * p * [ psi_f * i_q + (L_d - L_q) * i_d * i_q ]
        """
        p = self.params.pole_pairs
        psi_f = self.params.psi_f_wb
        ld = self.params.l_d_h
        lq = self.params.l_q_h

        torque_pm = psi_f * i_q
        torque_reluctance = (ld - lq) * i_d * i_q
        t_e = 1.5 * p * (torque_pm + torque_reluctance)
        return float(t_e)

    def execute_foc_step(
        self,
        target_torque_nm: float,
        i_a: float,
        i_b: float,
        i_c: float,
        rotor_theta_e_rad: float,
        dt_s: float = 0.0001  # 10 kHz (100 µs)
    ) -> Dict[str, Any]:
        """10 kHz FOC akım kontrol döngüsü."""
        # 1. İleri Clarke Dönüşümü (abc -> alpha, beta)
        i_alpha, i_beta = ClarkeTransform.forward(i_a, i_b, i_c)

        # 2. İleri Park Dönüşümü (alpha, beta -> d, q)
        i_d, i_q = ParkTransform.forward(i_alpha, i_beta, rotor_theta_e_rad)

        # 3. Hedef Akım Üretimi (MTPA / Basit Tork Orantısı)
        # MTPA basit yaklaşımı: i_d_ref = 0 (veya manyetik zayıflatma için negatif), i_q = Target / Kt
        kt = 1.5 * self.params.pole_pairs * self.params.psi_f_wb
        i_q_target = float(target_torque_nm / max(kt, 1e-3))
        i_d_target = 0.0  # Temel hız bölgesi

        # 4. Ayrık PI Akım Kontrolü (Gerilim Komutları V_d, V_q)
        v_d_cmd = self.pi_d.step(i_d_target, i_d, dt_s)
        v_q_cmd = self.pi_q.step(i_q_target, i_q, dt_s)

        # 5. Ters Park Dönüşümü (d, q -> alpha, beta)
        v_alpha, v_beta = ParkTransform.inverse(v_d_cmd, v_q_cmd, rotor_theta_e_rad)

        # 6. Ters Clarke Dönüşümü (alpha, beta -> a, b, c)
        v_a, v_b, v_c = ClarkeTransform.inverse(v_alpha, v_beta)

        # Üretilen anlık tork
        t_actual = self.compute_electromagnetic_torque(i_d, i_q)

        return {
            "i_alpha": i_alpha,
            "i_beta": i_beta,
            "i_d": i_d,
            "i_q": i_q,
            "i_d_target": i_d_target,
            "i_q_target": i_q_target,
            "v_d": v_d_cmd,
            "v_q": v_q_cmd,
            "v_alpha": v_alpha,
            "v_beta": v_beta,
            "v_a": v_a,
            "v_b": v_b,
            "v_c": v_c,
            "actual_torque_nm": t_actual,
            "target_torque_nm": target_torque_nm
        }
