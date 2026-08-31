"""
Tesla Rejeneratif Frenleme Birim Testleri (PyTest)
=================================================
Bu test paketi; Soğuk batarya ve yüksek SoC şarj kısıtlamalarını (SOP),
tork harmanlama (Blending) mantığını ve sıfır hızda Hold duruşunu test eder.

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

from src.tesla_rejeneratif_fren_yonetimi import (
    TeslaRegenerativeBrakeController,
    VehicleDynamicsState,
    StoppingMode
)


def test_soguk_hava_rejen_kisitlamasi():
    """0°C altındaki bataryada rejen çarpanının sıfırlandığı test edilir."""
    ctrl = TeslaRegenerativeBrakeController()
    limit = ctrl.compute_battery_charge_limit_factor(soc=0.70, temp_c=-5.0)
    assert limit == 0.0


def test_dolu_batarya_rejen_kisitlamasi():
    """%99 SoC dolu bataryada aşırı voltajı önlemek için rejenin sıfırlandığı test edilir."""
    ctrl = TeslaRegenerativeBrakeController()
    limit = ctrl.compute_battery_charge_limit_factor(soc=0.99, temp_c=25.0)
    assert limit == 0.0


def test_tork_harmanlama_onceligi():
    """Düşük fren talebinde sürtünme freninin sıfır kalıp sadece rejenin çalıştığı test edilir."""
    ctrl = TeslaRegenerativeBrakeController()
    state = VehicleDynamicsState(
        speed_kmh=60.0,
        accel_pedal_pct=0.0,
        brake_pedal_pct=0.0,
        battery_soc=0.60,
        battery_temp_c=25.0
    )

    out = ctrl.step_torque_blending(state, dt_s=0.01)
    assert out["regen_torque_nm"] > 0.0
    assert out["hydraulic_torque_nm"] == 0.0  # Balata kullanılmaz


def test_hold_modu_durus_kilidi():
    """Hız sıfırlandığında Hold modunun devreye girdiği test edilir."""
    ctrl = TeslaRegenerativeBrakeController()
    state = VehicleDynamicsState(speed_kmh=0.05, accel_pedal_pct=0.0)

    out = ctrl.step_torque_blending(state, dt_s=0.01)
    assert out["speed_kmh"] == 0.0
    assert out["hold_active"] is True
