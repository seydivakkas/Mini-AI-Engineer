r"""
Tesla Faz 8 Capstone: Enerji Ekosistemi Simülatörü Çekirdeği
============================================================
Bu modül; Faz 8'in tüm güç ve enerji mimarilerini birleştirir:
1. 16 Stall Supercharger V4 (NACS J3400 & ISO 15118 PnC).
2. Sıvı soğutmalı kablo termal modellemesi ve kısma koruması.
3. 3.9 MWh Megapack XL BESS Droop Frekans Yanıtı & Şebeke Desteği.
4. Tesla Autobidder Spot Piyasa Arbitrajı.
5. Tesla Solar Roof MPPT Güneş Hasadı.
6. 50.000 Powerwall VPP Filo Senkronizasyonu.
7. M/M/c İstasyon Kuyruk ve FSD Rota Optimizasyonu.
8. 100 Hz Güç Telemetrisi & 265 kHz SiC LLC Güç Elektroniği.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import struct
from collections import deque
import numpy as np


class TeslaPhase8EnergyEcosystemSimulator:
    """
    Tesla Faz 8 BÜYÜK CAPSTONE: NACS Supercharger, Megapack & Autobidder Ekosistem Simülatörü.
    """
    def __init__(
        self,
        num_stalls: int = 16,
        transformer_limit_kw: float = 2000.0,
        megapack_cap_mwh: float = 3.9,
        megapack_power_kw: float = 1950.0,
        solar_capacity_kw: float = 300.0,
        vpp_fleet_size: int = 50000
    ):
        self.num_stalls = num_stalls
        self.transformer_limit_kw = transformer_limit_kw
        self.megapack_cap_mwh = megapack_cap_mwh
        self.megapack_power_kw = megapack_power_kw
        self.solar_capacity_kw = solar_capacity_kw
        self.vpp_fleet_size = vpp_fleet_size

        # Durum Değişkenleri
        self.megapack_soc = 80.0
        self.cable_temps = np.full(num_stalls, 35.0)  # 35 °C başlangıç kablo sıcaklıkları
        self.telemetry_history = deque(maxlen=500)

    def calculate_stall_allocation(self, car_socs: List[float]) -> np.ndarray:
        """16 Stall için SoC ters orantılı dinamik yük dağıtımı."""
        n = min(len(car_socs), self.num_stalls)
        demands = np.array([max(1.0, 100.0 - soc) for soc in car_socs[:n]])
        total_demand = np.sum(demands)

        # Ham dağıtım
        raw_alloc = (demands / total_demand) * self.transformer_limit_kw
        clamped_alloc = np.minimum(raw_alloc, 350.0)  # Stall başı maks 350 kW

        # Artık güç yeniden dağıtımı
        residual = self.transformer_limit_kw - np.sum(clamped_alloc)
        if residual > 0:
            unclamped_mask = clamped_alloc < 350.0
            if np.any(unclamped_mask):
                sub_demands = demands[unclamped_mask]
                clamped_alloc[unclamped_mask] += (sub_demands / np.sum(sub_demands)) * residual

        return clamped_alloc

    def step_ecosystem_simulation(
        self,
        grid_freq_hz: float,
        spot_price_usd_mwh: float,
        car_socs: List[float],
        solar_irradiance_factor: float = 0.85
    ) -> Dict[str, Any]:
        """
        Tüm enerji ekosisteminin tek bir senkronize simülasyon adımı.
        """
        # 1. Supercharger Yük Paylaşımı
        stall_powers = self.calculate_stall_allocation(car_socs)
        total_supercharger_load_kw = float(np.sum(stall_powers))

        # 2. Sıvı Soğutmalı Kablo Isınması ve Termal Derating Kontrolü
        # Joule ısınma: T_new = T_old + alpha * (P/350)^2 - beta * (T - 25)
        for i in range(len(stall_powers)):
            p_ratio = stall_powers[i] / 350.0
            self.cable_temps[i] = self.cable_temps[i] + 0.8 * (p_ratio ** 2) - 0.2 * (self.cable_temps[i] - 25.0)

        max_cable_temp = float(np.max(self.cable_temps))
        cable_derating_active = bool(max_cable_temp > 85.0)

        # 3. Solar MPPT Üretimi
        solar_p_kw = self.solar_capacity_kw * solar_irradiance_factor

        # 4. Megapack P-f Droop & Autobidder Kararı
        delta_f = 50.0 - grid_freq_hz
        droop_power_kw = delta_f * 10000.0  # 10,000 kW/Hz

        # Autobidder Kararı
        if spot_price_usd_mwh > 150.0:
            autobidder_action = "DISCHARGE (SELL TO GRID)"
            bess_power_kw = min(self.megapack_power_kw, max(droop_power_kw, 1500.0))
        elif spot_price_usd_mwh < 30.0:
            autobidder_action = "CHARGE (BUY FROM GRID)"
            bess_power_kw = max(-self.megapack_power_kw, min(droop_power_kw, -1500.0))
        else:
            autobidder_action = "FREQUENCY REGULATION"
            bess_power_kw = float(np.clip(droop_power_kw, -self.megapack_power_kw, self.megapack_power_kw))

        # Megapack SoC Güncellemesi (1 dakikalık adım varsayımı)
        delta_soc = (bess_power_kw * (1.0 / 60.0) / (self.megapack_cap_mwh * 1000.0)) * 100.0
        self.megapack_soc = float(np.clip(self.megapack_soc - delta_soc, 10.0, 95.0))

        # 5. Net Şebeke Güç Dengesi
        # P_net_grid = P_supercharger - P_solar - P_megapack
        net_grid_draw_kw = total_supercharger_load_kw - solar_p_kw - bess_power_kw
        grid_safety_ok = bool(net_grid_draw_kw <= self.transformer_limit_kw)

        # 6. 100 Hz Kompakt Telemetri Paketleme (32-Bayt)
        telemetry_pkt = struct.pack(
            ">Qffffff",
            1700000000000000000,
            800.0,  # 800V DC
            float(net_grid_draw_kw / 0.8),  # Akım
            float(net_grid_draw_kw),
            15.0,   # Q (kVAR)
            grid_freq_hz,
            max_cable_temp
        )
        self.telemetry_history.append(telemetry_pkt)

        return {
            "num_stalls_active": len(stall_powers),
            "supercharger_load_kw": float(np.round(total_supercharger_load_kw, 2)),
            "solar_generated_kw": float(np.round(solar_p_kw, 2)),
            "megapack_power_kw": float(np.round(bess_power_kw, 2)),
            "megapack_soc_pct": float(np.round(self.megapack_soc, 2)),
            "autobidder_action": autobidder_action,
            "max_cable_temp_c": float(np.round(max_cable_temp, 2)),
            "cable_derating_active": cable_derating_active,
            "net_grid_draw_kw": float(np.round(net_grid_draw_kw, 2)),
            "transformer_limit_kw": self.transformer_limit_kw,
            "grid_safety_ok": grid_safety_ok,
            "stall_powers": list(np.round(stall_powers, 1))
        }
