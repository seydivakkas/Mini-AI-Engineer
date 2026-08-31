"""
Tesla FOC Motor Kontrolcü Birim Testleri (PyTest)
=================================================
Bu test paketi; Clarke ve Park ileri/ters dönüşümlerinin simetrisini,
elektromanyetik tork denklemini ve 10 kHz FOC döngüsünü test eder.

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

from src.tesla_foc_motor_kontrolcusu import (
    ClarkeTransform,
    ParkTransform,
    TeslaMotorParameters,
    TeslaFOCController
)


def test_clarke_ileri_ve_ters_donusum():
    """Clarke ve Ters Clarke dönüşümlerinin girdiyi birebir geri ürettiği test edilir."""
    i_a = 100.0
    i_b = -50.0
    i_c = -50.0

    alpha, beta = ClarkeTransform.forward(i_a, i_b, i_c)
    rec_a, rec_b, rec_c = ClarkeTransform.inverse(alpha, beta)

    assert pytest.approx(rec_a, 1e-4) == i_a
    assert pytest.approx(rec_b, 1e-4) == i_b
    assert pytest.approx(rec_c, 1e-4) == i_c


def test_park_ileri_ve_ters_donusum():
    """Park ve Ters Park dönüşümlerinin dönen açıyla birlikte simetrisini test eder."""
    alpha = 75.0
    beta = -30.0
    theta_e = np.pi / 3.0  # 60 derece

    d, q = ParkTransform.forward(alpha, beta, theta_e)
    rec_alpha, rec_beta = ParkTransform.inverse(d, q, theta_e)

    assert pytest.approx(rec_alpha, 1e-4) == alpha
    assert pytest.approx(rec_beta, 1e-4) == beta


def test_tork_hesabi_ve_foc_adimi():
    """Hedef tork komutu verildiğinde FOC kontrolcüsünün geçerli gerilim komutları ürettiği test edilir."""
    params = TeslaMotorParameters()
    foc = TeslaFOCController(params)

    # 200 Nm torka uygun faz akımları
    theta_e = 0.5
    i_d_test = 0.0
    i_q_test = 150.0
    # dq -> alpha beta
    alpha = i_d_test * np.cos(theta_e) - i_q_test * np.sin(theta_e)
    beta = i_d_test * np.sin(theta_e) + i_q_test * np.cos(theta_e)
    i_a, i_b, i_c = ClarkeTransform.inverse(alpha, beta)

    # 200 Nm tork komutu
    out = foc.execute_foc_step(
        target_torque_nm=200.0,
        i_a=i_a,
        i_b=i_b,
        i_c=i_c,
        rotor_theta_e_rad=theta_e,
        dt_s=0.0001
    )

    assert not np.isnan(out["v_a"])
    assert not np.isnan(out["v_b"])
    assert not np.isnan(out["v_c"])
    assert out["actual_torque_nm"] > 0.0
