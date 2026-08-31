"""
Tesla Supercharger V4 Birim Testleri (PyTest)
=============================================
Bu test paketi; şarj kablosu sıcaklığına göre akım kısma (Derating) fonksiyonunu,
termal diferansiyel denklemini ve 1000V DC şarj gücü hesaplamasını test eder.

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

from src.tesla_supercharger_v4_derater import TeslaSuperchargerV4CableDerater


def test_termal_derating_esikleri():
    """Farklı kablo sıcaklıklarında doğru akım kısıtlaması uygulandığı test edilir."""
    derater = TeslaSuperchargerV4CableDerater(nominal_current_a=500.0)

    # 1. 50 °C -> Tam güç (500 A)
    i_50, p_50, _ = derater.get_derated_charging_current(50.0)
    assert i_50 == 500.0
    assert p_50 == 500.0  # 500A * 1000V / 1000 = 500 kW

    # 2. 85 °C -> 375 A
    i_85, p_85, _ = derater.get_derated_charging_current(85.0)
    assert np.isclose(i_85, 375.0, atol=1.0)

    # 3. 90 °C -> 200 A
    i_90, p_90, _ = derater.get_derated_charging_current(90.0)
    assert i_90 == 200.0

    # 4. 98 °C -> Acil Kesme (0 A)
    i_98, p_98, _ = derater.get_derated_charging_current(98.0)
    assert i_98 == 0.0


def test_termal_adim_ve_joule_kaybi():
    """Joule kaybının ve kablo ısınmasının doğru hesaplandığı test edilir."""
    derater = TeslaSuperchargerV4CableDerater(cable_resistance_ohm=0.0012)
    res = derater.step_thermal_model(demanded_current_a=500.0, dt=1.0)

    # P_joule = 500^2 * 0.0012 = 300 W
    assert np.isclose(res["joule_loss_w"], 300.0, atol=1e-2)
    assert res["actual_current_a"] == 500.0


def test_sarj_seansi_simulasyonu():
    """120 saniyelik şarj seansının güvenli sıcaklıkta kaldığı test edilir."""
    derater = TeslaSuperchargerV4CableDerater()
    res = derater.simulate_charging_session(duration_s=60.0, demanded_current_a=500.0)

    assert len(res["zamanlar_s"]) == 120
    assert res["final_temp_c"] < 95.0
    assert res["final_power_kw"] > 0.0
