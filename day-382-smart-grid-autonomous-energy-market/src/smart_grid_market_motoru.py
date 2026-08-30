"""
Day 382: Smart Grid Autonomous Energy Balancing & Decentralized Agent Market
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Akıllı Elektrik Şebekesi (Smart Grid) Otonom Enerji Dengelemesini,
Çift Yönlü Çoklu-Ajan Açık Artırma Enerji Piyasasını (Double Auction),
Bölgesel Marjinal Fiyatlandırmayı (LMP) ve Frekans Kararlılık Kontrolünü (Swing Equation) simüle eder.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass, field


@dataclass
class GridBus:
    """Elektrik Şebekesi Düğüm (Bar) Modeli."""
    bus_id: int
    voltage_kv: float = 110.0
    base_load_mw: float = 20.0
    solar_gen_mw: float = 0.0
    wind_gen_mw: float = 0.0
    thermal_gen_mw: float = 0.0
    battery_soc_pct: float = 60.0
    battery_capacity_mwh: float = 40.0
    lmp_price_usd_per_mwh: float = 45.0


@dataclass
class TransmissionLine:
    """İletim Hattı Modeli."""
    from_bus: int
    to_bus: int
    reactance_x: float = 0.05
    capacity_mw: float = 80.0
    current_flow_mw: float = 0.0


@dataclass
class EnergyBid:
    """Enerji Piyasası Alış/Satış Teklifi."""
    agent_id: int
    bus_id: int
    is_producer: bool
    power_mw: float
    price_usd_mwh: float


class DoubleAuctionMarket:
    """
    Çoklu-Ajan Çift Yönlü Açık Artırma ve Piyasa Takas Fiyatı (MCP) Motoru.
    """
    def __init__(self):
        pass

    def clear_market(self, bids: List[EnergyBid]) -> Dict[str, Any]:
        producers = [b for b in bids if b.is_producer]
        consumers = [b for b in bids if not b.is_producer]

        producers.sort(key=lambda b: b.price_usd_mwh)
        consumers.sort(key=lambda b: -b.price_usd_mwh)

        total_traded_mw = 0.0
        mcp_price = 45.0
        matched_pairs = []

        prod_idx = 0
        cons_idx = 0

        while prod_idx < len(producers) and cons_idx < len(consumers):
            p_bid = producers[prod_idx]
            c_bid = consumers[cons_idx]

            if c_bid.price_usd_mwh >= p_bid.price_usd_mwh:
                trade_volume = min(p_bid.power_mw, c_bid.power_mw)
                mcp_price = (p_bid.price_usd_mwh + c_bid.price_usd_mwh) / 2.0
                total_traded_mw += trade_volume

                matched_pairs.append({
                    "producer_id": p_bid.agent_id,
                    "consumer_id": c_bid.agent_id,
                    "volume_mw": trade_volume,
                    "cleared_price": mcp_price
                })

                p_bid.power_mw -= trade_volume
                c_bid.power_mw -= trade_volume

                if p_bid.power_mw <= 1e-3:
                    prod_idx += 1
                if c_bid.power_mw <= 1e-3:
                    cons_idx += 1
            else:
                break

        return {
            "mcp_price_usd_per_mwh": round(mcp_price, 2),
            "total_traded_mw": round(total_traded_mw, 2),
            "num_matched_trades": len(matched_pairs),
            "market_liquidity_ratio": round(total_traded_mw / max(1.0, sum(b.power_mw for b in bids)), 3)
        }


class GridFrequencyStabilizer:
    """
    Şebeke Frekans Kararlılığı ve Salınım Denklemi (Swing Equation) Çözücüsü.
    M * d(delta_f)/dt + D * delta_f = P_gen - P_load
    """
    def __init__(self, inertia_m: float = 10.0, damping_d: float = 2.5, nominal_freq_hz: float = 50.0):
        self.M = inertia_m
        self.D = damping_d
        self.nominal_f = nominal_freq_hz
        self.current_delta_f = 0.0

    def step_frequency(self, power_mismatch_mw: float, dt_sec: float = 0.1) -> float:
        """
        Salınım denklemi üzerinden frekans türevini çözer ve droop geri beslemesi uygular.
        """
        scale = 0.001
        d_df = (power_mismatch_mw * scale - self.D * self.current_delta_f) / self.M
        self.current_delta_f += d_df * dt_sec
        # Droop hızlı sönümleme
        self.current_delta_f *= 0.90
        return self.nominal_f + self.current_delta_f


class SmartGridSimulation:
    """
    14-Baralı Akıllı Şebeke, Yenilenebilir Entegrasyon ve Dağıtık Piyasa Simülasyonu.
    """
    def __init__(self, num_buses: int = 14):
        self.num_buses = num_buses
        self.market = DoubleAuctionMarket()
        self.stabilizer = GridFrequencyStabilizer()

        self.buses: List[GridBus] = []
        for i in range(num_buses):
            base_l = 15.0 + np.random.uniform(5.0, 20.0)
            solar = 12.0 + np.random.uniform(0.0, 15.0) if i % 2 == 0 else 0.0
            wind = 10.0 + np.random.uniform(0.0, 18.0) if i % 3 == 0 else 0.0
            thermal = 25.0 if i in [0, 3, 7] else 0.0
            bus = GridBus(
                bus_id=i, 
                base_load_mw=base_l, 
                solar_gen_mw=solar, 
                wind_gen_mw=wind, 
                thermal_gen_mw=thermal
            )
            self.buses.append(bus)

        self.lines: List[TransmissionLine] = []
        for i in range(num_buses - 1):
            line = TransmissionLine(from_bus=i, to_bus=i + 1, capacity_mw=60.0)
            self.lines.append(line)
        self.lines.append(TransmissionLine(from_bus=num_buses - 1, to_bus=0, capacity_mw=60.0))

    def step_grid_time_step(self, hour_index: int = 12) -> Dict[str, Any]:
        solar_factor = max(0.0, np.sin((hour_index - 6) * np.pi / 12)) if 6 <= hour_index <= 18 else 0.0
        wind_factor = 0.5 + 0.5 * np.sin(hour_index * 0.4)

        bids: List[EnergyBid] = []
        total_gen_mw = 0.0
        total_load_mw = 0.0

        for bus in self.buses:
            actual_solar = bus.solar_gen_mw * solar_factor
            actual_wind = bus.wind_gen_mw * wind_factor
            actual_thermal = bus.thermal_gen_mw

            net_local = (actual_solar + actual_wind + actual_thermal) - bus.base_load_mw
            if net_local > 5.0 and bus.battery_soc_pct < 95.0:
                bus.battery_soc_pct = min(95.0, bus.battery_soc_pct + 1.5)
                net_local -= 5.0
            elif net_local < -5.0 and bus.battery_soc_pct > 20.0:
                bus.battery_soc_pct = max(20.0, bus.battery_soc_pct - 1.5)
                net_local += 5.0

            total_gen_mw += (actual_solar + actual_wind + actual_thermal)
            total_load_mw += bus.base_load_mw

            if net_local > 0:
                price = 25.0 if (actual_solar + actual_wind) > actual_thermal else 55.0
                bids.append(EnergyBid(agent_id=bus.bus_id, bus_id=bus.bus_id, is_producer=True, power_mw=net_local, price_usd_mwh=price))
            else:
                bids.append(EnergyBid(agent_id=bus.bus_id, bus_id=bus.bus_id, is_producer=False, power_mw=abs(net_local), price_usd_mwh=65.0))

        market_res = self.market.clear_market(bids)
        mcp = market_res["mcp_price_usd_per_mwh"]

        for line in self.lines:
            line.current_flow_mw = min(line.capacity_mw * 0.9, np.random.uniform(15.0, 45.0))

        for bus in self.buses:
            congestion_adder = 8.0 if bus.bus_id in [4, 9] else 0.0
            bus.lmp_price_usd_per_mwh = mcp + congestion_adder

        power_mismatch = total_gen_mw - total_load_mw
        current_freq_hz = self.stabilizer.step_frequency(power_mismatch)
        freq_dev_hz = abs(current_freq_hz - 50.0)

        renewable_gen_mw = sum(b.solar_gen_mw * solar_factor + b.wind_gen_mw * wind_factor for b in self.buses)
        renewable_penetration_pct = (renewable_gen_mw / max(1.0, total_gen_mw)) * 100.0

        return {
            "total_gen_mw": round(total_gen_mw, 1),
            "total_load_mw": round(total_load_mw, 1),
            "renewable_penetration_pct": round(renewable_penetration_pct, 1),
            "mcp_price_usd_mwh": mcp,
            "grid_frequency_hz": round(current_freq_hz, 4),
            "frequency_deviation_hz": round(freq_dev_hz, 4),
            "avg_battery_soc_pct": round(float(np.mean([b.battery_soc_pct for b in self.buses])), 1),
            "buses_lmp": [b.lmp_price_usd_per_mwh for b in self.buses],
            "lines_flow": [l.current_flow_mw for l in self.lines]
        }


class SmartGridBenchmark:
    """
    Akıllı Şebeke ve Enerji Piyasası Başarım Paketi.
    """
    def __init__(self):
        self.sim = SmartGridSimulation(num_buses=14)

    def run_benchmark(self, num_hours: int = 24) -> Dict[str, Any]:
        np.random.seed(42)
        freq_devs = []
        renewables = []
        mcps = []
        socs = []

        for h in range(num_hours):
            res = self.sim.step_grid_time_step(hour_index=h % 24)
            freq_devs.append(res["frequency_deviation_hz"])
            renewables.append(res["renewable_penetration_pct"])
            mcps.append(res["mcp_price_usd_mwh"])
            socs.append(res["avg_battery_soc_pct"])

        return {
            "num_hours": num_hours,
            "avg_frequency_deviation_hz": float(np.mean(freq_devs)),
            "max_frequency_deviation_hz": float(np.max(freq_devs)),
            "avg_renewable_penetration_pct": float(np.mean(renewables)),
            "avg_mcp_usd_mwh": float(np.mean(mcps)),
            "avg_battery_soc_pct": float(np.mean(socs)),
            "grid_stability_pct": max(0.0, 100.0 - float(np.mean(freq_devs)) * 200.0),
            "sample_step": res
        }

    def kos(self, num_hours: int = 24) -> Dict[str, Any]:
        return self.run_benchmark(num_hours)
