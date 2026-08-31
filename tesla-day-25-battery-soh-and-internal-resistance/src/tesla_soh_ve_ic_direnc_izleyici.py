"""
Tesla Batarya Sağlık Durumu (SoH) ve Çevrimiçi İç Direnç İzleyici
================================================================
Bu modül; Kapasite Kaybı (Capacity Fade), İç Direnç Artışı (Resistance Growth),
Recursive Least Squares (RLS) ile çevrimiçi parametre kestirimi ve
SEI tabakası yaşlanma (Cycle & Calendar Aging) modellerini gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


def calculate_soh_capacity(current_capacity_ah: float, fresh_capacity_ah: float = 75.0) -> float:
    """Kapasite tabanlı SoH (State of Health) yüzdesi hesaplar."""
    if fresh_capacity_ah <= 0.0:
        return 0.0
    return float(np.clip((current_capacity_ah / fresh_capacity_ah) * 100.0, 0.0, 100.0))


def calculate_soh_resistance(current_r0_ohm: float, fresh_r0_ohm: float = 0.0015, eol_multiplier: float = 2.0) -> float:
    """
    Direnç tabanlı SoH yüzdesi hesaplar.
    EOL (End of Life): İç direncin 2 katına (fresh * 2.0) çıkması durumu %0 SoH kabul edilir.
    """
    eol_r0 = fresh_r0_ohm * eol_multiplier
    if eol_r0 <= fresh_r0_ohm:
        return 100.0
    soh_r = (eol_r0 - current_r0_ohm) / (eol_r0 - fresh_r0_ohm) * 100.0
    return float(np.clip(soh_r, 0.0, 100.0))


class RecursiveLeastSquaresR0:
    """
    Unutma Faktörlü Özyinelemeli En Küçük Kareler (RLS with Forgetting Factor).
    Model: y = phi * theta -> (V_ocv - V_t) = I * R0
    theta = [R0]
    """
    def __init__(self, initial_r0_guess: float = 0.0015, lambda_forgetting: float = 0.995):
        self.theta = float(initial_r0_guess)
        self.P = 1.0  # Başlangıç kovaryansı
        self.lam = float(lambda_forgetting)

    def update(self, delta_i: float, delta_v: float) -> float:
        """
        delta_i = I - I_prev
        delta_v = (V_ocv - V_t) - (V_ocv_prev - V_t_prev)
        """
        phi = delta_i
        # Kazanç K = P * phi / (lambda + phi^2 * P)
        denom = self.lam + (phi ** 2) * self.P
        if abs(denom) < 1e-9:
            return self.theta

        K = (self.P * phi) / denom
        # Artık Hata (Error)
        error = delta_v - phi * self.theta
        # Parametre Güncellemesi
        self.theta += K * error
        # Kovaryans Güncellemesi
        self.P = (self.P - K * phi * self.P) / self.lam
        return float(self.theta)


class BatteryCycleAgingSimulator:
    """
    Tesla 4680 / 2170 Hücre Yaşlanma Modeli (SEI Katmanı Büyümesi).
    Q_loss = B * exp(-Ea / (R*T)) * (Ah_throughput)^0.5
    """
    def __init__(self, fresh_capacity_ah: float = 75.0, fresh_r0_ohm: float = 0.0015):
        self.fresh_capacity_ah = fresh_capacity_ah
        self.fresh_r0_ohm = fresh_r0_ohm
        self.current_capacity_ah = fresh_capacity_ah
        self.current_r0_ohm = fresh_r0_ohm
        self.total_cycles = 0

    def step_cycles(self, cycle_count: int, temp_c: float = 35.0, dod_depth_of_discharge: float = 0.80):
        """Hücreyi belirli sayıda şarj/deşarj döngüsü boyunca yaşlandırır."""
        self.total_cycles += cycle_count
        # SEI Büyümesi: Döngü sayısının kareköküyle orantılı kapasite kaybı
        t_k = temp_c + 273.15
        arrhenius_factor = np.exp(-31700.0 / (8.314 * t_k)) * 1e5
        
        # 1000 döngüde yaklaşık %10-15 kayıp
        cap_fade_pct = 0.0035 * arrhenius_factor * (dod_depth_of_discharge ** 0.8) * np.sqrt(self.total_cycles)
        self.current_capacity_ah = float(self.fresh_capacity_ah * max(1.0 - cap_fade_pct, 0.60))

        # İç direnç artışı (SEI direnci)
        r0_increase_pct = 0.0055 * arrhenius_factor * np.sqrt(self.total_cycles)
        self.current_r0_ohm = float(self.fresh_r0_ohm * (1.0 + r0_increase_pct))

    def get_health_status(self) -> Dict[str, float]:
        soh_c = calculate_soh_capacity(self.current_capacity_ah, self.fresh_capacity_ah)
        soh_r = calculate_soh_resistance(self.current_r0_ohm, self.fresh_r0_ohm)
        # Bileşik SoH (En kısıtlayıcı olan)
        soh_combined = min(soh_c, soh_r)
        return {
            "total_cycles": float(self.total_cycles),
            "capacity_ah": self.current_capacity_ah,
            "r0_ohm": self.current_r0_ohm,
            "soh_capacity_pct": soh_c,
            "soh_resistance_pct": soh_r,
            "soh_combined_pct": soh_combined
        }
