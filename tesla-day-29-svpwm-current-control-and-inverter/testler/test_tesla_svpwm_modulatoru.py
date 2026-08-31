"""
Tesla SVPWM Modülatör Birim Testleri (PyTest)
============================================
Bu test paketi; 6 sektörün doğru tespitini, T1+T2+T0 süre korunumunu ve
görev çevrimlerinin (Duty Cycles) [0, 1] aralığında kaldığını test eder.

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

from src.tesla_svpwm_modulatoru import TeslaSVPWMModulator


def test_sektor_tespiti():
    """Farklı açılardaki gerilim vektörlerinin doğru sektöre düştüğü test edilir."""
    mod = TeslaSVPWMModulator(v_dc_bus=400.0)

    # 30 derece -> Sektör 1 (0-60°)
    sec1, _, _, _ = mod.compute_sector_and_times(v_alpha=100.0, v_beta=57.7)
    assert sec1 == 1

    # 90 derece -> Sektör 2 (60-120°)
    sec2, _, _, _ = mod.compute_sector_and_times(v_alpha=0.0, v_beta=100.0)
    assert sec2 == 2

    # 180 derece -> Sektör 4 (180-240°)
    sec4, _, _, _ = mod.compute_sector_and_times(v_alpha=-100.0, v_beta=0.0)
    assert sec4 == 4


def test_sure_korunumu_ve_toplami():
    """T1 + T2 + T0 toplamının PWM periyoduna (100 µs) eşit olduğu test edilir."""
    mod = TeslaSVPWMModulator(v_dc_bus=400.0, switching_freq_hz=10000.0)

    for deg in range(0, 360, 45):
        rad = np.radians(deg)
        v_a = 150.0 * np.cos(rad)
        v_b = 150.0 * np.sin(rad)
        _, t1, t2, t0 = mod.compute_sector_and_times(v_a, v_b)
        assert pytest.approx(t1 + t2 + t0, 1e-6) == mod.t_pwm_s


def test_gorev_cevrimi_sinirlari():
    """da, db, dc görev çevrimlerinin daima [0.0, 1.0] aralığında olduğu test edilir."""
    mod = TeslaSVPWMModulator(v_dc_bus=400.0)

    for deg in range(0, 360, 30):
        rad = np.radians(deg)
        out = mod.compute_phase_duty_cycles(200.0 * np.cos(rad), 200.0 * np.sin(rad))
        assert 0.0 <= out["duty_a"] <= 1.0
        assert 0.0 <= out["duty_b"] <= 1.0
        assert 0.0 <= out["duty_c"] <= 1.0
