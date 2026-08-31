"""
Tesla Octovalve Termal Yönetim ve Isı Pompası Kontrolcüsü
=========================================================
Bu modül; Tesla Model Y / 3 araçlarında bulunan 8-Yollu Döner Valf (Octovalve)
ve Isı Pompası (Heat Pump) termal döngülerini yöneterek batarya ön koşullandırma
(Preconditioning), kabin ısıtma/soğutma ve motor kayıp ısısı geri kazanımını sağlar.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class OctovalveMode(Enum):
    CABIN_HEATING_AMBIENT = 1       # Dış havadan kabine ısı pompası
    BATTERY_PRECONDITION_HEAT = 2   # Motor + Isı Pompası ile bataryayı 45°C'ye ısıtma (Supercharger)
    BATTERY_ACTIVE_COOLING = 3      # Chiller ile bataryayı soğutma (Yüksek yük / Sıcak hava)
    POWERTRAIN_HEAT_HARVEST = 4     # İnvertör ve motor ısısını batarya/kabine aktarma
    MAX_DEFROST = 5                 # Cam buğu çözme ve maksimum kabin ısıtma


@dataclass
class VehicleThermalState:
    temp_battery_c: float = 20.0
    temp_cabin_c: float = 18.0
    temp_powertrain_c: float = 35.0
    temp_ambient_c: float = 5.0
    compressor_power_w: float = 0.0
    coolant_flow_lpm: float = 15.0
    heat_pump_cop: float = 3.5


class TeslaOctovalveController:
    """
    Tesla 8-Yollu Octovalve ve Isı Pompası Termal Kontrolcüsü.
    """
    def __init__(self, target_battery_temp_c: float = 45.0, target_cabin_temp_c: float = 22.0):
        self.target_battery_temp = target_battery_temp_c
        self.target_cabin_temp = target_cabin_temp_c
        # Batarya Isıl Kapasitesi: 500 kg LFP/NMC paket ~ 500,000 J/K
        self.c_th_battery = 450000.0  # J/K
        self.c_th_cabin = 25000.0     # J/K

    def determine_mode(self, state: VehicleThermalState, supercharge_target_set: bool = False) -> OctovalveMode:
        """Sensör sıcaklıklarına ve sürüş hedefine göre en verimli modu seçer."""
        if supercharge_target_set and state.temp_battery_c < (self.target_battery_temp - 2.0):
            return OctovalveMode.BATTERY_PRECONDITION_HEAT
        elif state.temp_battery_c > 42.0 and not supercharge_target_set:
            return OctovalveMode.BATTERY_ACTIVE_COOLING
        elif state.temp_powertrain_c > 45.0 and state.temp_cabin_c < self.target_cabin_temp:
            return OctovalveMode.POWERTRAIN_HEAT_HARVEST
        elif state.temp_cabin_c < (self.target_cabin_temp - 3.0):
            return OctovalveMode.CABIN_HEATING_AMBIENT
        else:
            return OctovalveMode.POWERTRAIN_HEAT_HARVEST

    def step(self, state: VehicleThermalState, mode: OctovalveMode, dt_s: float = 1.0) -> Dict[str, float]:
        """
        1 saniyelik termal diferansiyel denklem çözümü:
        C_th * dT/dt = Q_in - Q_out
        """
        # Motorun ürettiği sürekli kayıp ısısı: ~2.5 kW
        q_powertrain_w = 2500.0
        q_battery_net_w = 0.0
        q_cabin_net_w = 0.0

        if mode == OctovalveMode.BATTERY_PRECONDITION_HEAT:
            # Kompresör + Motor ısısı bataryaya yönlendirilir
            state.compressor_power_w = 3500.0
            state.heat_pump_cop = 3.2
            # Isı pompasından üretilen termal güç = COP * W_el
            q_heat_pump = state.compressor_power_w * state.heat_pump_cop
            # Toplam ısı bataryaya basılır
            q_battery_net_w = q_heat_pump + q_powertrain_w
            # Çevreye hafif ısı kaybı
            q_battery_net_w -= (state.temp_battery_c - state.temp_ambient_c) * 45.0

        elif mode == OctovalveMode.BATTERY_ACTIVE_COOLING:
            state.compressor_power_w = 2500.0
            state.heat_pump_cop = 2.8
            # Chiller bataryadan ısı çeker
            q_cooling = state.compressor_power_w * state.heat_pump_cop
            q_battery_net_w = -q_cooling + 500.0  # Hücre iç ısınması düşülür

        elif mode == OctovalveMode.CABIN_HEATING_AMBIENT:
            state.compressor_power_w = 1500.0
            state.heat_pump_cop = 3.8
            q_cabin_net_w = state.compressor_power_w * state.heat_pump_cop
            # Kabin çevreye ısı kaybeder
            q_cabin_net_w -= (state.temp_cabin_c - state.temp_ambient_c) * 80.0

        elif mode == OctovalveMode.POWERTRAIN_HEAT_HARVEST:
            # Motor ısısı kabine ve bataryaya dengeli dağıtılır
            state.compressor_power_w = 500.0
            q_battery_net_w = q_powertrain_w * 0.5 - (state.temp_battery_c - state.temp_ambient_c) * 30.0
            q_cabin_net_w = q_powertrain_w * 0.5 - (state.temp_cabin_c - state.temp_ambient_c) * 60.0

        # Diferansiyel sıcaklık değişimi: dT = (Q_net * dt) / C_th
        d_temp_batt = (q_battery_net_w * dt_s) / self.c_th_battery
        d_temp_cabin = (q_cabin_net_w * dt_s) / self.c_th_cabin

        state.temp_battery_c += d_temp_batt
        state.temp_cabin_c += d_temp_cabin

        return {
            "temp_battery_c": float(state.temp_battery_c),
            "temp_cabin_c": float(state.temp_cabin_c),
            "q_battery_net_w": float(q_battery_net_w),
            "q_cabin_net_w": float(q_cabin_net_w),
            "compressor_power_w": float(state.compressor_power_w),
            "cop": float(state.heat_pump_cop)
        }
