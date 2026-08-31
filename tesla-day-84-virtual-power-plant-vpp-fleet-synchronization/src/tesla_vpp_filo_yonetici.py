r"""
Tesla Sanal Enerji Santrali (VPP) Filo Yönetim Çekirdeği
=========================================================
Bu modül; on binlerce ev tipi Tesla Powerwall (13.5 kWh / 5.0 kW) bataryasını
tek bir Sanal Enerji Santrali (VPP) olarak birleştirir, şebeke acil durum
güç taleplerini (örneğin 150 MW) dağıtık olarak orkestre eder ve kullanıcı
yedekleme rezervini (%20) garanti altına alır.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaVirtualPowerPlantFleet:
    """
    Tesla Sanal Enerji Santrali (VPP) Filo Orkestratörü.
    """
    def __init__(
        self,
        fleet_size: int = 50000,
        unit_capacity_kwh: float = 13.5,
        unit_max_power_kw: float = 5.0,
        reserve_soc_pct: float = 20.0,
        random_seed: int = 42
    ):
        self.fleet_size = fleet_size
        self.unit_capacity_kwh = unit_capacity_kwh
        self.unit_max_power_kw = unit_max_power_kw
        self.reserve_soc_pct = reserve_soc_pct

        np.random.seed(random_seed)
        # Powerwall filosu SoC dağılımı (%50 ile %95 arası)
        self.soc_array = np.random.uniform(50.0, 95.0, size=fleet_size)
        self.max_power_array = np.full(fleet_size, unit_max_power_kw)

    def get_available_fleet_capacity_mw(self) -> float:
        """Kullanılabilir aktif deşarj kapasitesini MW cinsinden hesaplar."""
        eligible_mask = self.soc_array > self.reserve_soc_pct
        total_kw = np.sum(self.max_power_array[eligible_mask])
        return float(total_kw / 1000.0)

    def dispatch_grid_demand(self, demand_mw: float, duration_hours: float = 1.0) -> Dict[str, Any]:
        """
        Şebekeden gelen güç talebini (MW) filodaki uygun Powerwall ünitelerine dağıtır.
        """
        demand_kw = demand_mw * 1000.0
        eligible_mask = self.soc_array > self.reserve_soc_pct
        num_eligible = int(np.sum(eligible_mask))

        if num_eligible == 0:
            return {
                "demand_mw": demand_mw,
                "dispatched_mw": 0.0,
                "demand_met": False,
                "eligible_units": 0,
                "avg_unit_power_kw": 0.0
            }

        # İstenen güç birim başına düşen miktar
        required_kw_per_unit = demand_kw / num_eligible
        actual_kw_per_unit = min(self.unit_max_power_kw, required_kw_per_unit)

        dispatched_total_kw = actual_kw_per_unit * num_eligible
        dispatched_mw = dispatched_total_kw / 1000.0

        # SoC güncellemesi
        energy_discharged_kwh = actual_kw_per_unit * duration_hours
        delta_soc = (energy_discharged_kwh / self.unit_capacity_kwh) * 100.0

        # Sadece katılan ünitelerin SoC'sini düşür
        self.soc_array[eligible_mask] = np.maximum(
            self.reserve_soc_pct,
            self.soc_array[eligible_mask] - delta_soc
        )

        return {
            "fleet_size": self.fleet_size,
            "demand_mw": demand_mw,
            "dispatched_mw": float(np.round(dispatched_mw, 2)),
            "demand_met": bool(dispatched_mw >= (demand_mw - 1e-3)),
            "eligible_units": num_eligible,
            "avg_unit_power_kw": float(np.round(actual_kw_per_unit, 2)),
            "avg_fleet_soc_pct": float(np.round(np.mean(self.soc_array), 2))
        }
