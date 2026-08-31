"""
Tesla Uzay Vektör Darbe Genişlik Modülasyonu (SVPWM) ve İnvertör Sürücüsü
=========================================================================
Bu modül; 2-seviyeli 3-fazlı gerilim kaynaklı invertör (VSI) için
Uzay Vektör PWM (Space Vector PWM - SVPWM) sektör tespiti, aktif vektör
süreleri (T1, T2, T0), 7-segment simetrik anahtarlama ve ölü zaman (Dead-time)
korumasını gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaSVPWMModulator:
    """
    Tesla Uzay Vektör Modülatörü (SVPWM Generator).
    """
    def __init__(self, v_dc_bus: float = 400.0, switching_freq_hz: float = 10000.0, dead_time_us: float = 1.5):
        self.v_dc = v_dc_bus
        self.f_sw = switching_freq_hz
        self.t_pwm_s = 1.0 / switching_freq_hz  # 10 kHz -> 100 µs
        self.dead_time_s = dead_time_us * 1e-6

    def compute_sector_and_times(self, v_alpha: float, v_beta: float) -> Tuple[int, float, float, float]:
        """
        Sektör tespiti (1-6) ve T1, T2, T0 anahtarlama sürelerinin hesabı.
        """
        # Vektör açısı (0 ile 2*pi arasında)
        angle_rad = np.arctan2(v_beta, v_alpha)
        if angle_rad < 0:
            angle_rad += 2.0 * np.pi

        # Sektör 1: 0 - 60°, Sektör 2: 60 - 120°, ..., Sektör 6: 300 - 360°
        sector = int(angle_rad // (np.pi / 3.0)) + 1
        sector = int(np.clip(sector, 1, 6))

        # Sektör içi yerel açı (0 ile pi/3 arasında)
        theta_local = angle_rad - (sector - 1) * (np.pi / 3.0)

        # Referans gerilim genliği
        v_ref_mag = np.sqrt(v_alpha**2 + v_beta**2)
        v_max_linear = self.v_dc / np.sqrt(3.0)  # ~230.9 V

        # Aşırı modülasyonu sınırla (Linear Modulation Region)
        if v_ref_mag > v_max_linear:
            v_ref_mag = v_max_linear

        # T1 ve T2 süreleri (Standart SVPWM formülleri)
        t1 = (np.sqrt(3.0) * self.t_pwm_s * v_ref_mag / self.v_dc) * np.sin(np.pi / 3.0 - theta_local)
        t2 = (np.sqrt(3.0) * self.t_pwm_s * v_ref_mag / self.v_dc) * np.sin(theta_local)
        t0 = self.t_pwm_s - t1 - t2

        t1 = float(max(0.0, t1))
        t2 = float(max(0.0, t2))
        t0 = float(max(0.0, t0))

        return sector, t1, t2, t0

    def compute_phase_duty_cycles(self, v_alpha: float, v_beta: float) -> Dict[str, Any]:
        """
        7-Segment Simetrik Merkezli (Center-Aligned) Görev Çevrimleri (Duty Cycles da, db, dc).
        """
        sector, t1, t2, t0 = self.compute_sector_and_times(v_alpha, v_beta)

        # Sektörlere göre faz açılış süreleri (Ta, Tb, Tc)
        if sector == 1:
            ta = t1 + t2 + t0 / 2.0
            tb = t2 + t0 / 2.0
            tc = t0 / 2.0
        elif sector == 2:
            ta = t1 + t0 / 2.0
            tb = t1 + t2 + t0 / 2.0
            tc = t0 / 2.0
        elif sector == 3:
            ta = t0 / 2.0
            tb = t1 + t2 + t0 / 2.0
            tc = t2 + t0 / 2.0
        elif sector == 4:
            ta = t0 / 2.0
            tb = t1 + t0 / 2.0
            tc = t1 + t2 + t0 / 2.0
        elif sector == 5:
            ta = t2 + t0 / 2.0
            tb = t0 / 2.0
            tc = t1 + t2 + t0 / 2.0
        else:  # Sektör 6
            ta = t1 + t2 + t0 / 2.0
            tb = t0 / 2.0
            tc = t1 + t0 / 2.0

        duty_a = float(np.clip(ta / self.t_pwm_s, 0.0, 1.0))
        duty_b = float(np.clip(tb / self.t_pwm_s, 0.0, 1.0))
        duty_c = float(np.clip(tc / self.t_pwm_s, 0.0, 1.0))

        # Ölü Zaman (Dead-time) eklenmiş güvenli aktif süreler
        dead_time_duty_penalty = self.dead_time_s / self.t_pwm_s

        return {
            "sector": sector,
            "t1_us": t1 * 1e6,
            "t2_us": t2 * 1e6,
            "t0_us": t0 * 1e6,
            "duty_a": duty_a,
            "duty_b": duty_b,
            "duty_c": duty_c,
            "v_dc_bus": self.v_dc,
            "dead_time_penalty_pct": dead_time_duty_penalty * 100.0
        }
