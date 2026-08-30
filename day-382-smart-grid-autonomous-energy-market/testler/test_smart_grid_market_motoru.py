"""
Day 382: Unit Tests for Smart Grid Autonomous Energy Balancing & Decentralized Agent Market
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from smart_grid_market_motoru import (
    GridBus,
    TransmissionLine,
    EnergyBid,
    DoubleAuctionMarket,
    GridFrequencyStabilizer,
    SmartGridSimulation,
    SmartGridBenchmark
)


def test_double_auction_market_clearing():
    """Çift yönlü açık artırma piyasa takasının doğru eşleştiğini test eder."""
    market = DoubleAuctionMarket()
    bids = [
        EnergyBid(agent_id=1, bus_id=0, is_producer=True, power_mw=20.0, price_usd_mwh=30.0),
        EnergyBid(agent_id=2, bus_id=1, is_producer=True, power_mw=15.0, price_usd_mwh=40.0),
        EnergyBid(agent_id=3, bus_id=2, is_producer=False, power_mw=25.0, price_usd_mwh=60.0)
    ]

    res = market.clear_market(bids)
    assert res["total_traded_mw"] == 25.0
    assert res["num_matched_trades"] > 0
    assert 30.0 <= res["mcp_price_usd_per_mwh"] <= 60.0


def test_swing_equation_frequency_stability():
    """Salınım denklemi ve droop kontrolcüsünün frekansı dengede tuttuğunu test eder."""
    stab = GridFrequencyStabilizer(nominal_freq_hz=50.0)
    # Güç fazlalığı durumunda frekans hafifçe yükselir
    f_high = stab.step_frequency(power_mismatch_mw=50.0)
    assert f_high > 50.0

    # Güç açığı durumunda frekans hafifçe düşer
    f_low = stab.step_frequency(power_mismatch_mw=-80.0)
    assert f_low < 50.0
    assert abs(f_low - 50.0) < 0.5, "Frekans sapması tolerans dahilinde olmalıdır."


def test_bess_battery_arbitrage_soc():
    """Güneşli saatlerde BESS bataryanın şarj olduğunu ve SoC sınırlarını koruduğunu test eder."""
    sim = SmartGridSimulation(num_buses=6)
    # Öğlen 12:00 (Yüksek güneş)
    res_noon = sim.step_grid_time_step(hour_index=12)
    assert res_noon["renewable_penetration_pct"] > 0.0
    assert 10.0 <= res_noon["avg_battery_soc_pct"] <= 100.0


def test_tam_smart_grid_benchmark_ve_stability():
    """24 saatlik tam akıllı şebeke ve piyasa benchmarkını test eder."""
    bench = SmartGridBenchmark()
    res = bench.kos(num_hours=24)

    assert res["num_hours"] == 24
    assert res["avg_frequency_deviation_hz"] < 0.1, "Ortalama frekans sapması 0.1 Hz altında olmalıdır."
    assert res["avg_renewable_penetration_pct"] > 20.0
    assert res["grid_stability_pct"] >= 80.0
