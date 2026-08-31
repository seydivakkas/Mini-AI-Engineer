r"""
Tesla Supercharger V4 Sıvı Soğutmalı Kablo ve Termal Kısma (Derating) Çekirdeği
================================================================================
Bu modül; Tesla Supercharger V4 istasyonlarının 1000V DC mimarisini, glikol
sıvı soğutmalı şarj kablosunun Joule ısınma diferansiyel modelini ($P=I^2 R$)
ve kablo sıcaklığına bağlı dinamik akım kısma (Derating) kalkanını gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaSuperchargerV4CableDerater:
    """
    Tesla Supercharger V4 Termal ve Akım Kısma (Derating) Yöneticisi.
    """
    def __init__(
        self,
        nominal_current_a: float = 500.0,
        voltage_v: float = 1000.0,
        cable_resistance_ohm: float = 0.0012,  # 1.2 mOhm
        coolant_temp_c: float = 25.0,
        thermal_mass_j_per_k: float = 450.0,
        heat_transfer_w_per_k: float = 15.0
    ):
        self.nominal_current = nominal_current_a
        self.voltage = voltage_v
        self.r_cable = cable_resistance_ohm
        self.t_coolant = coolant_temp_c
        self.mc_p = thermal_mass_j_per_k
        self.ha = heat_transfer_w_per_k

        self.cable_temp_c = coolant_temp_c

    def get_derated_charging_current(self, cable_temp_c: float) -> Tuple[float, float, str]:
        """
        Kablo sıcaklığına göre izin verilen maksimum akımı ve gücü hesaplar.
        """
        if cable_temp_c > 95.0:
            return 0.0, 0.0, "ACİL TERMAL KESME (Kablonun Erimesi Engellendi)"
        elif cable_temp_c > 85.0:
            allowed_i = self.nominal_current * 0.40  # 200 A (%60 kisinti)
            reason = "AĞIR TERMAL KISINTI (%40 Akım / 200 kW)"
        elif cable_temp_c > 70.0:
            # 70 C ile 85 C arası doğrusal kısma
            alpha = (cable_temp_c - 70.0) / 15.0
            allowed_i = self.nominal_current * (1.0 - 0.25 * alpha)  # 500A -> 375A
            reason = "ORTA SEVİYE TERMAL DERATING"
        else:
            allowed_i = self.nominal_current
            reason = "TAM GÜÇ (NOMINAL 500 kW / 500A)"

        power_kw = (allowed_i * self.voltage) / 1000.0
        return allowed_i, power_kw, reason

    def step_thermal_model(self, demanded_current_a: float, dt: float = 0.1) -> Dict[str, Any]:
        """
        Tek bir zaman adımında kablo ısınmasını ve akım kısıtlamasını simüle eder.
        """
        allowed_i, power_kw, reason = self.get_derated_charging_current(self.cable_temp_c)
        actual_i = min(demanded_current_a, allowed_i)

        # Joule Kaybı: P_joule = I^2 * R
        p_joule = (actual_i ** 2) * self.r_cable

        # Sıvı Soğutmaya Isı Transferi: P_cool = hA * (T_cable - T_coolant)
        p_cooling = self.ha * (self.cable_temp_c - self.t_coolant)

        # Sıcaklık Değişimi: dT/dt = (P_joule - P_cooling) / (m * cp)
        dt_cable = (p_joule - p_cooling) / self.mc_p
        self.cable_temp_c += dt_cable * dt

        return {
            "cable_temp_c": self.cable_temp_c,
            "actual_current_a": actual_i,
            "charging_power_kw": (actual_i * self.voltage) / 1000.0,
            "joule_loss_w": p_joule,
            "status_reason": reason
        }

    def simulate_charging_session(
        self,
        duration_s: float = 120.0,
        demanded_current_a: float = 500.0,
        dt: float = 0.5
    ) -> Dict[str, Any]:
        """Tam bir ultra hızlı şarj seansını simüle eder."""
        steps = int(duration_s / dt)
        zamanlar = []
        sicakliklar = []
        akimlar = []
        gucler = []

        for i in range(steps):
            t = i * dt
            res = self.step_thermal_model(demanded_current_a, dt=dt)
            zamanlar.append(t)
            sicakliklar.append(res["cable_temp_c"])
            akimlar.append(res["actual_current_a"])
            gucler.append(res["charging_power_kw"])

        return {
            "zamanlar_s": zamanlar,
            "sicakliklar_c": sicakliklar,
            "akimlar_a": akimlar,
            "gucler_kw": gucler,
            "final_temp_c": self.cable_temp_c,
            "final_power_kw": gucler[-1]
        }
