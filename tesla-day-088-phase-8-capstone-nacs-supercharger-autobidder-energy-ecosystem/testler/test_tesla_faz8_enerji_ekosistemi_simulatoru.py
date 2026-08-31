"""
Tesla Faz 8 Capstone Birim Testleri (PyTest)
============================================
Bu test paketi; 16-Stall Supercharger yük paylaşımını, Megapack BESS ve
Solar Roof entegrasyonunu ve şebeke trafo güvenlik kalkanını test eder.

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

from src.tesla_faz8_enerji_ekosistemi_simulatoru import TeslaPhase8EnergyEcosystemSimulator


def test_16_stall_yuk_paylasimi():
    """16 stall için toplam yükün trafo sınırını (2000 kW) aşmadığı test edilir."""
    sim = TeslaPhase8EnergyEcosystemSimulator(num_stalls=16, transformer_limit_kw=2000.0)
    car_socs = [10.0 + i * 5.0 for i in range(16)]

    stall_powers = sim.calculate_stall_allocation(car_socs)

    assert len(stall_powers) == 16
    assert np.isclose(np.sum(stall_powers), 2000.0, atol=1.0)
    assert np.all(stall_powers <= 350.0)


def test_ekosistem_sebeke_guvenlik_kalkani():
    """Solar ve Megapack desteğiyle net şebeke çekişinin güvenli kaldığı test edilir."""
    sim = TeslaPhase8EnergyEcosystemSimulator()
    car_socs = [20.0] * 16

    res = sim.step_ecosystem_simulation(
        grid_freq_hz=49.9,
        spot_price_usd_mwh=160.0,
        car_socs=car_socs,
        solar_irradiance_factor=0.9
    )

    assert res["grid_safety_ok"] is True
    assert res["net_grid_draw_kw"] <= sim.transformer_limit_kw
    assert res["solar_generated_kw"] > 0.0
    assert res["megapack_power_kw"] > 0.0  # Şebekeye deşarj desteği


def test_kablo_sicaklik_guvenligi():
    """Sıvı soğutmalı şarj kablolarının sıcaklıklarının güvenli bölgede kaldığı test edilir."""
    sim = TeslaPhase8EnergyEcosystemSimulator()
    car_socs = [30.0] * 16

    res = sim.step_ecosystem_simulation(
        grid_freq_hz=50.0,
        spot_price_usd_mwh=80.0,
        car_socs=car_socs
    )

    assert res["max_cable_temp_c"] < 95.0
    assert res["cable_derating_active"] is False
