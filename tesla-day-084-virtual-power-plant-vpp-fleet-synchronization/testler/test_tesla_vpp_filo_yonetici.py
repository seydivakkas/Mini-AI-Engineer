"""
Tesla VPP Filo Birim Testleri (PyTest)
======================================
Bu test paketi; 50.000 Powerwall ünitesinin toplam deşarj kapasitesini,
150 MW şebeke talebinin dağıtılmasını ve kullanıcı rezerv kilidini test eder.

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

from src.tesla_vpp_filo_yonetici import TeslaVirtualPowerPlantFleet


def test_filo_kapasitesi():
    """50.000 Powerwall'un toplam deşarj kapasitesinin 250 MW olduğu test edilir."""
    fleet = TeslaVirtualPowerPlantFleet(fleet_size=50000, unit_max_power_kw=5.0)
    cap_mw = fleet.get_available_fleet_capacity_mw()

    assert np.isclose(cap_mw, 250.0, atol=1.0)


def test_sebeke_talep_karsilama():
    """150 MW talebin filoya eşit ve güvenli dağıtıldığı test edilir."""
    fleet = TeslaVirtualPowerPlantFleet(fleet_size=50000, unit_max_power_kw=5.0)
    res = fleet.dispatch_grid_demand(demand_mw=150.0, duration_hours=1.0)

    assert res["demand_met"] is True
    assert np.isclose(res["dispatched_mw"], 150.0, atol=1.0)
    assert np.isclose(res["avg_unit_power_kw"], 3.0, atol=0.2)


def test_kullanici_rezerv_korumasi():
    """Kullanıcı %20 rezervinin altına hiçbir ünitenin düşürülmediği test edilir."""
    fleet = TeslaVirtualPowerPlantFleet(fleet_size=1000, reserve_soc_pct=20.0)
    # Tüm üniteleri %25 SoC'ye ayarla ve aşırı deşarj iste
    fleet.soc_array = np.full(1000, 25.0)
    res = fleet.dispatch_grid_demand(demand_mw=5.0, duration_hours=2.0)

    # SoC en az %20 olmalı
    assert np.all(fleet.soc_array >= 20.0)
