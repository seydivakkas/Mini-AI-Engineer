"""
Tesla Batarya Hücre Dengeleme Kontrolcüsü (Passive & Active Balancing)
=====================================================================
Bu modül; 96S batarya modülündeki hücre voltaj uyumsuzluklarını (Imbalance)
Pasif Direnç Dengeleme (Passive Bleeding) ve Aktif Endüktif Enerji Aktarımı
(Active Bidirectional Shuttling) yöntemleriyle sıfırlayan kontrolcüleri gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class BalancingStrategy(Enum):
    PASSIVE_BLEEDING = "PASSIVE_BLEEDING"
    ACTIVE_INDUCTIVE = "ACTIVE_INDUCTIVE"


@dataclass
class BatteryCell:
    cell_id: int
    voltage_v: float
    capacity_ah: float = 75.0
    soc: float = 0.80
    temperature_c: float = 25.0
    bleed_switch_active: bool = False

    def update_voltage_from_soc(self):
        # 3.0V - 4.2V NMC eğrisi
        soc_c = np.clip(self.soc, 0.001, 0.999)
        self.voltage_v = float(3.0 + 1.20 * soc_c + 0.05 * np.log(soc_c) - 0.02 * np.exp(-15.0 * soc_c))


class TeslaBalancingController:
    """
    Tesla BMS Hücre Dengeleme Yöneticisi.
    """
    def __init__(
        self,
        strategy: BalancingStrategy = BalancingStrategy.PASSIVE_BLEEDING,
        voltage_threshold_mv: float = 10.0,
        bleed_resistor_ohm: float = 33.0,
        active_transfer_current_a: float = 2.0,
        active_efficiency: float = 0.88,
        max_bleed_temp_c: float = 55.0
    ):
        self.strategy = strategy
        self.voltage_threshold_v = voltage_threshold_mv / 1000.0
        self.r_bleed = bleed_resistor_ohm
        self.active_current = active_transfer_current_a
        self.active_eff = active_efficiency
        self.max_temp_c = max_bleed_temp_c

    def step_balancing(self, cells: List[BatteryCell], dt_s: float = 1.0) -> Dict[str, Any]:
        voltages = [c.voltage_v for c in cells]
        min_v = min(voltages)
        max_v = max(voltages)
        imbalance_v = max_v - min_v

        total_heat_dissipated_w = 0.0
        active_transfers_count = 0

        # Eğer dengesizlik eşik değerinin altındaysa dengeleme yapma
        if imbalance_v <= self.voltage_threshold_v:
            for c in cells:
                c.bleed_switch_active = False
            return {
                "imbalance_mv": imbalance_v * 1000.0,
                "balancing_active": False,
                "heat_w": 0.0,
                "min_v": min_v,
                "max_v": max_v,
                "active_switches": 0
            }

        # 1. PASİF DİRENÇ DENGELEME (Bleeding Resistors)
        if self.strategy == BalancingStrategy.PASSIVE_BLEEDING:
            target_v = min_v + self.voltage_threshold_v
            active_count = 0

            for c in cells:
                # Sıcaklık koruması ve voltaj hedefi
                if c.voltage_v > target_v and c.temperature_c < self.max_temp_c:
                    c.bleed_switch_active = True
                    active_count += 1
                    # Direnç üzerinden akım akıtılır: I = V / R
                    i_bleed = c.voltage_v / self.r_bleed
                    # Kapasite azaltma: dSoC = -(I * dt) / (Q * 3600)
                    c.soc -= (i_bleed * dt_s) / (c.capacity_ah * 3600.0)
                    c.update_voltage_from_soc()
                    total_heat_dissipated_w += (c.voltage_v ** 2) / self.r_bleed
                else:
                    c.bleed_switch_active = False

            return {
                "imbalance_mv": (max([c.voltage_v for c in cells]) - min([c.voltage_v for c in cells])) * 1000.0,
                "balancing_active": True,
                "heat_w": total_heat_dissipated_w,
                "min_v": min([c.voltage_v for c in cells]),
                "max_v": max([c.voltage_v for c in cells]),
                "active_switches": active_count
            }

        # 2. AKTİF ÇİFT YÖNLÜ ENDÜKTİF DENGELEME (Active Bidirectional Charge Shuttling)
        elif self.strategy == BalancingStrategy.ACTIVE_INDUCTIVE:
            max_cell_idx = int(np.argmax(voltages))
            min_cell_idx = int(np.argmin(voltages))

            high_cell = cells[max_cell_idx]
            low_cell = cells[min_cell_idx]

            # Yüksek hücreden akım çekilir
            high_cell.soc -= (self.active_current * dt_s) / (high_cell.capacity_ah * 3600.0)
            high_cell.update_voltage_from_soc()

            # Düşük hücreye verimle aktarılır
            delivered_current = self.active_current * self.active_eff
            low_cell.soc += (delivered_current * dt_s) / (low_cell.capacity_ah * 3600.0)
            low_cell.update_voltage_from_soc()

            # Kayıp güç
            loss_w = (self.active_current * high_cell.voltage_v) * (1.0 - self.active_eff)
            total_heat_dissipated_w += loss_w
            active_transfers_count = 1

            return {
                "imbalance_mv": (max([c.voltage_v for c in cells]) - min([c.voltage_v for c in cells])) * 1000.0,
                "balancing_active": True,
                "heat_w": total_heat_dissipated_w,
                "min_v": min([c.voltage_v for c in cells]),
                "max_v": max([c.voltage_v for c in cells]),
                "active_switches": 2
            }
