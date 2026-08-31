r"""
Tesla Megapack BESS ve Şebeke Frekans (Droop) Kontrol Çekirdeği
===============================================================
Bu modül; Tesla 3.9 MWh Megapack XL ve Powerwall 3 batarya enerji depolama
sistemlerinin (BESS) şebeke oluşturan (Grid-Forming) invertör mimarisini,
Aktif Güç - Frekans ($P-f$) ve Reaktif Güç - Gerilim ($Q-V$) Droop
kontrol döngüsünü gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaMegapackBESSController:
    """
    Tesla Megapack ve Powerwall Şebeke Frekans / Gerilim Droop Kontrolcüsü.
    """
    def __init__(
        self,
        capacity_mwh: float = 3.9,
        max_power_mw: float = 1.95,
        nominal_freq_hz: float = 50.0,
        droop_gain_kw_per_hz: float = 10000.0,
        nominal_voltage_v: float = 400.0,
        droop_gain_q_kvar_per_v: float = 50.0,
        initial_soc_pct: float = 75.0
    ):
        self.capacity_kwh = capacity_mwh * 1000.0
        self.max_power_kw = max_power_mw * 1000.0
        self.f_nominal = nominal_freq_hz
        self.k_droop = droop_gain_kw_per_hz
        self.v_nominal = nominal_voltage_v
        self.k_q = droop_gain_q_kvar_per_v

        self.soc_pct = initial_soc_pct

    def compute_active_droop_power(self, grid_freq_hz: float) -> Tuple[float, str]:
        """
        Şebeke frekans sapmasına göre aktif güç enjeksiyonu veya emilimini hesaplar.
        Pozitif değer: Şebekeye Güç Verme (Deşarj)
        Negatif değer: Şebekeden Güç Çekme (Şarj)
        """
        freq_error = self.f_nominal - grid_freq_hz
        p_raw = freq_error * self.k_droop

        # Batarya SoC ve Güç Limitleri
        if p_raw > 0:  # Deşarj Talebi
            if self.soc_pct <= 10.0:
                return 0.0, "BATARYA BOŞ (%10 Emniyet Sınırı)"
            p_actual = min(p_raw, self.max_power_kw)
            action = f"ŞEBEKEYE GÜÇ ENJEKSİYONU ({p_actual:.1f} kW Deşarj)"
        elif p_raw < 0:  # Şarj Talebi (Aşırı Frekans)
            if self.soc_pct >= 95.0:
                return 0.0, "BATARYA DOLU (%95 Şarj Sınırı)"
            p_actual = max(p_raw, -self.max_power_kw)
            action = f"ŞEBEKEDEN FAZLA GÜCÜ EMME ({abs(p_actual):.1f} kW Şarj)"
        else:
            p_actual = 0.0
            action = "ŞEBEKE NOMİNAL (50.0 Hz Standby)"

        return p_actual, action

    def compute_reactive_droop_power(self, grid_voltage_v: float) -> Tuple[float, str]:
        """Gerilim sapmasına göre reaktif güç (kVAR) desteği hesaplar."""
        v_error = self.v_nominal - grid_voltage_v
        q_raw = v_error * self.k_q
        q_actual = float(np.clip(q_raw, -self.max_power_kw * 0.5, self.max_power_kw * 0.5))
        return q_actual, f"Reaktif Destek: {q_actual:.1f} kVAR"

    def step_bess_simulation(
        self,
        grid_freq_hz: float,
        grid_voltage_v: float = 400.0,
        dt_s: float = 0.1
    ) -> Dict[str, Any]:
        """Tek bir zaman adımında Megapack tepkisini ve SoC değişimini simüle eder."""
        p_kw, action_p = self.compute_active_droop_power(grid_freq_hz)
        q_kvar, action_q = self.compute_reactive_droop_power(grid_voltage_v)

        # Enerji Değişimi: dE = P * dt (kWh)
        energy_kwh = (p_kw * dt_s) / 3600.0
        delta_soc = (energy_kwh / self.capacity_kwh) * 100.0
        self.soc_pct = float(np.clip(self.soc_pct - delta_soc, 0.0, 100.0))

        return {
            "grid_freq_hz": grid_freq_hz,
            "grid_voltage_v": grid_voltage_v,
            "active_power_kw": p_kw,
            "reactive_power_kvar": q_kvar,
            "soc_pct": self.soc_pct,
            "action": action_p
        }
