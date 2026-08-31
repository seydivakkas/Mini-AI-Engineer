"""
Tesla Autobidder Birim Testleri (PyTest)
========================================
Bu test paketi; spot elektrik fiyatına göre alım/satım kararlarını,
batarya yıpranma maliyetini ve 24 saatlik arbitraj karlılığını test eder.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import numpy as np
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_autobidder_ticaret_motoru import TeslaAutobidderTrader


def test_autobidder_karar_mantigi():
    """Yüksek fiyatta satış, düşük fiyatta alış kararı verildiği test edilir."""
    trader = TeslaAutobidderTrader(high_price_threshold_usd=150.0, low_price_threshold_usd=30.0)

    # 1. 200 $/MWh -> Satış
    action_sell, p_sell = trader.decide_trade_action(200.0, soc_pct=70.0)
    assert "DISCHARGE_TO_GRID" in action_sell
    assert p_sell > 0.0

    # 2. 20 $/MWh -> Alış
    action_buy, p_buy = trader.decide_trade_action(20.0, soc_pct=30.0)
    assert "CHARGE_FROM_GRID" in action_buy
    assert p_buy < 0.0

    # 3. 80 $/MWh -> Standby
    action_std, p_std = trader.decide_trade_action(80.0, soc_pct=50.0)
    assert "STANDBY_OPTIMAL" in action_std
    assert p_std == 0.0


def test_24_saatlik_arbitraj_ve_net_kar():
    """24 saatlik simülasyonda pozitif net arbitraj karı elde edildiği test edilir."""
    trader = TeslaAutobidderTrader()
    # Gece ucuz, akşam pahalı fiyat profili
    prices = [20.0]*6 + [60.0]*10 + [220.0]*4 + [50.0]*4
    res = trader.simulate_24h_trading(prices)

    assert res["total_revenue_usd"] > 0.0
    assert res["total_charging_cost_usd"] > 0.0
    assert res["net_profit_usd"] > 0.0
    assert res["final_soc_pct"] >= 10.0


def test_batarya_koruma_sinirlari():
    """Batarya %10 altına düşmeyecek şekilde deşarjın sınırlandığı test edilir."""
    trader = TeslaAutobidderTrader(initial_soc_pct=15.0)
    # Çok pahalı fiyat ama batarya neredeyse boş
    action, p = trader.decide_trade_action(300.0, soc_pct=10.0)

    assert action == "STANDBY_OPTIMAL" or p == 0.0
