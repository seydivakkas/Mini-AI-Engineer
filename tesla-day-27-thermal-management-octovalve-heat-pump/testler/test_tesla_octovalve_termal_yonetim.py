"""
Tesla Octovalve Termal Yönetim Birim Testleri (PyTest)
======================================================
Bu test paketi; Octovalve çalışma modlarının doğru seçildiğini, batarya
ön ısıtma diferansiyel denklemlerini ve COP enerji verimini test eder.

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

from src.tesla_octovalve_termal_yonetim import (
    TeslaOctovalveController,
    VehicleThermalState,
    OctovalveMode
)


def test_octovalve_mod_secimi():
    """Supercharger hedefi seçildiğinde batarya ön ısıtma modunun tetiklendiği test edilir."""
    ctrl = TeslaOctovalveController()
    state = VehicleThermalState(temp_battery_c=15.0, temp_cabin_c=20.0)

    mode = ctrl.determine_mode(state, supercharge_target_set=True)
    assert mode == OctovalveMode.BATTERY_PRECONDITION_HEAT


def test_batarya_asiri_sicak_sogutma_modu():
    """Batarya 42°C üzerine çıktığında aktif soğutma moduna geçildiği test edilir."""
    ctrl = TeslaOctovalveController()
    state = VehicleThermalState(temp_battery_c=45.0, temp_cabin_c=22.0)

    mode = ctrl.determine_mode(state, supercharge_target_set=False)
    assert mode == OctovalveMode.BATTERY_ACTIVE_COOLING


def test_batarya_on_isitma_sicaklik_artisi():
    """Ön ısıtma modunda 60 saniyede batarya sıcaklığının pozitif yönde arttığı test edilir."""
    ctrl = TeslaOctovalveController()
    state = VehicleThermalState(temp_battery_c=10.0, temp_ambient_c=0.0)

    init_temp = state.temp_battery_c
    for _ in range(60):
        ctrl.step(state, OctovalveMode.BATTERY_PRECONDITION_HEAT, dt_s=1.0)

    assert state.temp_battery_c > init_temp
