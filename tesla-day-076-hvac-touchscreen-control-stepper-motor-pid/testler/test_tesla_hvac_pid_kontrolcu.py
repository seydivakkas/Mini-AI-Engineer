"""
Tesla HVAC PID Kontrolcü Birim Testleri (PyTest)
================================================
Bu test paketi; step motor darbe hesaplamasını, PID tek adım çıktısını
ve kabin soğutma yörüngesi kararlılığını test eder.

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

from src.tesla_hvac_pid_kontrolcu import TeslaHVACPIDController


def test_step_motor_darbe_hesabi():
    """1.8 derece/adım ile hedef açıların doğru darbe sayısına dönüştüğü test edilir."""
    ctrl = TeslaHVACPIDController()

    assert ctrl.calculate_stepper_pulses(0.0) == 0
    assert ctrl.calculate_stepper_pulses(18.0) == 10
    assert ctrl.calculate_stepper_pulses(-36.0) == -20


def test_pid_tek_adim_sogutma_ciktisi():
    """Sıcak kabinde (35 °C) PID kontrolcüsünün pozitif soğutma gücü ürettiği test edilir."""
    ctrl = TeslaHVACPIDController(initial_temp_c=35.0, target_temp_c=21.5)
    res = ctrl.step()

    assert res["cooling_power_pct"] > 0.0
    assert res["error_c"] > 0.0
    assert res["current_temp_c"] <= 35.0


def test_kabin_sogutma_ve_kararlilik():
    """60 saniyelik simülasyonda sıcaklığın hedefe doğru düştüğü test edilir."""
    ctrl = TeslaHVACPIDController(initial_temp_c=35.0, target_temp_c=21.5)
    res = ctrl.simulate_cooling_trajectory(duration_s=60.0)

    assert res["final_temp_c"] < 35.0
    assert len(res["zamanlar_s"]) == 600
    assert res["sicakliklar_c"][-1] < res["sicakliklar_c"][0]
