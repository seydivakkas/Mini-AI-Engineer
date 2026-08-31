"""
Tesla Rejeneratif Frenleme ve Tork Harmanlama (Torque Blending) Yönetimi
========================================================================
Bu modül; Tek Pedallı Sürüş (One-Pedal Drive), Rejeneratif Elektrikli Fren
ile Hidrolik Sürtünme Freni harmanlamasını (Blending) ve Batarya SoC /
Sıcaklık (SOP) şarj kabul sınırlarını gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class StoppingMode(Enum):
    HOLD = "HOLD"    # 0 km/h'de otomatik elektrikli park freni / motor tutma
    ROLL = "ROLL"    # Serbest süzülme (Boş vites gibi)
    CREEP = "CREEP"  # Klasik otomatik vites gibi hafif ileri kayma (5 km/h)


@dataclass
class VehicleDynamicsState:
    speed_kmh: float = 80.0
    accel_pedal_pct: float = 0.0     # %0 = Ayak gazdan çekildi (Tam Rejen)
    brake_pedal_pct: float = 0.0     # %0 = Fren basılı değil
    battery_soc: float = 0.70        # %70 SoC
    battery_temp_c: float = 25.0     # 25°C İdeal
    vehicle_mass_kg: float = 1850.0  # Tesla Model 3
    wheel_radius_m: float = 0.34


class TeslaRegenerativeBrakeController:
    """
    Tesla Rejenerasyon ve Tork Harmanlama Kontrolcüsü.
    """
    def __init__(self, max_regen_torque_nm: float = 300.0, max_regen_power_kw: float = 75.0):
        self.max_regen_torque = max_regen_torque_nm
        self.max_regen_power_w = max_regen_power_kw * 1000.0
        self.stopping_mode = StoppingMode.HOLD

    def compute_battery_charge_limit_factor(self, soc: float, temp_c: float) -> float:
        """
        Batarya Şarj Kabul Çarpanı (0.0 ile 1.0 arası).
        - Soğuk Batarya (< 0°C): Lityum kaplama (Plating) önlemek için 0 kW!
        - Dolu Batarya (> %95 SoC): Aşırı gerilim önlemek için kısılır.
        """
        # Sıcaklık faktörü
        if temp_c < 0.0:
            temp_factor = 0.0
        elif temp_c < 15.0:
            temp_factor = temp_c / 15.0
        else:
            temp_factor = 1.0

        # SoC faktörü
        if soc > 0.98:
            soc_factor = 0.0
        elif soc > 0.85:
            soc_factor = (0.98 - soc) / (0.98 - 0.85)
        else:
            soc_factor = 1.0

        return float(np.clip(temp_factor * soc_factor, 0.0, 1.0))

    def step_torque_blending(self, state: VehicleDynamicsState, dt_s: float = 0.01) -> Dict[str, Any]:
        """
        100 Hz Tork Harmanlama ve Yavaşlama Hesabı.
        """
        # 1. Batarya Şarj Kabul Sınırı
        charge_limit = self.compute_battery_charge_limit_factor(state.battery_soc, state.battery_temp_c)
        available_regen_torque = self.max_regen_torque * charge_limit

        # 2. Gaz Pedalından Talep Edilen Rejenerasyon (Tek Pedallı Sürüş)
        # Gaz pedalı %0 ise tam rejen talep edilir
        if state.accel_pedal_pct == 0.0 and state.speed_kmh > 0.5:
            demanded_regen_torque = self.max_regen_torque
        elif state.accel_pedal_pct < 20.0 and state.speed_kmh > 0.5:
            # %0-%20 arası geçiş bölgesi
            demanded_regen_torque = self.max_regen_torque * ((20.0 - state.accel_pedal_pct) / 20.0)
        else:
            demanded_regen_torque = 0.0

        # 3. Fren Pedalından Talep Edilen İlave Fren Torku
        demanded_brake_pedal_torque = (state.brake_pedal_pct / 100.0) * 1200.0  # Max 1200 Nm sürtünme

        total_demanded_brake_torque = demanded_regen_torque + demanded_brake_pedal_torque

        # 4. Tork Harmanlama (Blending)
        # Öncelik Rejenerasyondadır:
        actual_regen_torque = min(total_demanded_brake_torque, available_regen_torque)
        # Kalan açık hidrolik sürtünme freniyle karşılanır:
        actual_hydraulic_torque = max(0.0, total_demanded_brake_torque - actual_regen_torque)

        # 5. Geri Kazanılan Elektriksel Güç (P = T * omega)
        v_ms = state.speed_kmh / 3.6
        omega_wheel = v_ms / state.wheel_radius_m
        regen_power_w = actual_regen_torque * omega_wheel * 0.90  # %90 invertör/motor verimi
        regen_power_w = min(regen_power_w, self.max_regen_power_w)

        # 6. Hız ve Dinamik Güncelleme (F = m * a)
        total_retard_force = (actual_regen_torque + actual_hydraulic_torque) / state.wheel_radius_m
        decel_ms2 = total_retard_force / state.vehicle_mass_kg

        # Hız güncelleme
        new_v_ms = max(0.0, v_ms - decel_ms2 * dt_s)
        state.speed_kmh = float(new_v_ms * 3.6)

        # 0 km/h Hold Durumu
        hold_engaged = False
        if state.speed_kmh <= 0.1 and self.stopping_mode == StoppingMode.HOLD:
            hold_engaged = True
            state.speed_kmh = 0.0

        return {
            "speed_kmh": state.speed_kmh,
            "regen_torque_nm": float(actual_regen_torque),
            "hydraulic_torque_nm": float(actual_hydraulic_torque),
            "regen_power_kw": float(regen_power_w / 1000.0),
            "charge_limit_factor": charge_limit,
            "deceleration_g": float(decel_ms2 / 9.81),
            "hold_active": hold_engaged
        }
