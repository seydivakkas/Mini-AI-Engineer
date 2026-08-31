"""
Tesla Optimus ZMP Denge Birim Testleri (PyTest)
===============================================
Bu test paketi; LIPM ZMP hesabını, destek poligonu kararlılığını
ve Capture Point itme kurtarma stratejilerini test eder.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_optimus_zmp_denge_kontrolcu import TeslaOptimusZMPBalanceController


def test_zmp_hesaplama():
    """LIPM ZMP noktasının doğru hesaplandığı test edilir."""
    ctrl = TeslaOptimusZMPBalanceController()
    # x_com = 0.05 m, x_ddot = 0.981 m/s^2 -> x_zmp = 0.05 - (0.85 / 9.81) * 0.981 = 0.05 - 0.085 = -0.035 m
    x_zmp, y_zmp = ctrl.compute_zmp(x_com=0.05, y_com=0.0, x_ddot_com=0.981, y_ddot_com=0.0)

    assert pytest.approx(x_zmp, 1e-3) == -0.035
    assert y_zmp == 0.0


def test_destek_poligonu_guvenligi():
    """Destek poligonu içi ve dışı noktaların doğru sınıflandırıldığı test edilir."""
    ctrl = TeslaOptimusZMPBalanceController()

    # Destek içi (Güvenli)
    assert ctrl.is_zmp_within_support(0.02, 0.05) is True

    # Destek dışı (Güvensiz - Devrilme tehlikesi)
    assert ctrl.is_zmp_within_support(0.35, 0.0) is False
    assert ctrl.is_zmp_within_support(0.0, 0.40) is False


def test_itme_kurtarma_ve_capture_point():
    """Dışarıdan gelen güçlü bir itmede adım atma (Stepping) stratejisi tetiklenmelidir."""
    ctrl = TeslaOptimusZMPBalanceController()

    # 1. Küçük İtme (15 Ns) -> Adım gerekmez (Bilek stratejisi)
    res_small = ctrl.push_recovery_step(0.0, 0.0, 0.0, 0.0, ext_impulse_ns=15.0)
    assert res_small["step_required"] is False

    # 2. Büyük İtme (55 Ns) -> Adım atma zorunlu (Stepping)
    res_large = ctrl.push_recovery_step(0.0, 0.0, 0.0, 0.0, ext_impulse_ns=55.0)
    assert res_large["step_required"] is True
    assert "STEPPING_STRATEGY" in res_large["recovery_strategy"]
