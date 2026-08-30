"""
Day 394: Unit Tests for Microsecond Algorithmic Trading with LOB Dynamics
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from hft_lob_trading_motoru import (
    LimitOrderBook,
    HawkesOrderFlowGenerator,
    AlmgrenChrissExecutor,
    HFTTradingBenchmark
)


def test_limit_order_book_micro_price():
    """LOB hacim ağırlıklı mikro-fiyat hesaplamasını test eder."""
    lob = LimitOrderBook(initial_mid_price=100.0, tick_size=0.01)
    micro_p = lob.compute_micro_price()

    # Mikro fiyat en iyi alış ve satış arasında olmalıdır
    bb, ba, _, _ = lob.get_best_bid_ask()
    assert bb <= micro_p <= ba
    assert abs(micro_p - 100.0) < 0.05


def test_hawkes_intensity_clustering():
    """Hawkes nokta sürecinin taban yoğunluk üstünde patlamalar ürettiğini test eder."""
    hawkes = HawkesOrderFlowGenerator(mu=100.0, alpha=800.0, beta=1000.0)
    intensity = hawkes.compute_intensity(dt_s=1e-4, current_intensity=100.0)

    assert intensity >= 100.0
    assert hawkes.branching_ratio < 1.0, "Kararlı nokta süreci için eta < 1 olmalıdır."


def test_almgren_chriss_liquidation_trajectory():
    """Almgren-Chriss tasfiye eğrisinin zamanla monoton azalarak sıfıra ulaştığını test eder."""
    executor = AlmgrenChrissExecutor(total_shares=10000.0, time_horizon_s=10.0)
    x_0 = executor.get_optimal_trajectory(t=0.0)
    x_mid = executor.get_optimal_trajectory(t=5.0)
    x_end = executor.get_optimal_trajectory(t=10.0)

    assert abs(x_0 - 10000.0) < 1.0
    assert 0.0 < x_mid < x_0
    assert abs(x_end) < 1.0


def test_tam_hft_trading_benchmark():
    """Tam mikrosaniye HFT algoritmik ticaret benchmarkını test eder."""
    bench = HFTTradingBenchmark(num_ticks=2000)
    res = bench.kos()

    assert res["num_ticks"] == 2000
    assert res["sharpe_ratio"] > 3.0
    assert res["max_drawdown_pct"] < 2.0
    assert res["avg_latency_us"] < 5.0
