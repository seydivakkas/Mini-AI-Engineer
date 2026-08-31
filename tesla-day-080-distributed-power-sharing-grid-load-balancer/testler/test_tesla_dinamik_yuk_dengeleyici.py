"""
Tesla Dinamik Yük Dengeleyici Birim Testleri (PyTest)
=====================================================
Bu test paketi; 8 stall'luk Supercharger istasyonunda trafo aşım korumasını,
bireysel stall sınırlarını ve SoC orantılı dinamik güç paylaşımını test eder.

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

from src.tesla_dinamik_yuk_dengeleyici import TeslaDynamicLoadBalancer


def test_trafo_kapasitesi_asimi_engelleme():
    """8 araç tam güç istese dahi toplam gücün 1000 kW trafo sınırını aşmadığı test edilir."""
    balancer = TeslaDynamicLoadBalancer(grid_capacity_kw=1000.0, max_stall_power_kw=250.0)
    socs = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]  # Hepsi çok aç
    res = balancer.balance_power(socs)

    assert res["total_allocated_kw"] <= 1000.0
    assert res["overload_prevented"] is True


def test_soc_ters_oranti_onceliklendirmesi():
    """Düşük SoC'ye sahip aracın yüksek SoC'li araca göre daha fazla güç aldığı test edilir."""
    balancer = TeslaDynamicLoadBalancer(grid_capacity_kw=400.0, max_stall_power_kw=250.0)
    socs = [15.0, 85.0]  # Biri %15, diğeri %85
    res = balancer.balance_power(socs)

    p_low_soc = res["allocated_powers_kw"][0]
    p_high_soc = res["allocated_powers_kw"][1]

    assert p_low_soc > p_high_soc


def test_bireysel_stall_limiti():
    """Tek bir araca 250 kW stall limitinden fazla güç verilmediği test edilir."""
    balancer = TeslaDynamicLoadBalancer(grid_capacity_kw=1000.0, max_stall_power_kw=250.0)
    socs = [5.0]  # Sadece tek araç bağlı ve trafo 1000 kW
    res = balancer.balance_power(socs)

    assert res["allocated_powers_kw"][0] <= 250.0
