"""
Tesla EKF SoC Kestirici Birim Testleri (PyTest)
===============================================
Bu test paketi; Coulomb Counting'in kayma açığını, EKF'nin başlangıç hatasını
hızlıca düzeltmesini (Convergence) ve kovaryans güncellemelerini test eder.

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

from src.tesla_ekf_soc_kestirici import BatteryEKFSoCEstimator, CoulombCounter


def test_coulomb_counting_bias_kaymasi():
    """DC akım yanlılığı altında Coulomb Counting'in zamanla sürüklendiği test edilir."""
    cc = CoulombCounter(initial_soc=0.80, capacity_ah=10.0)
    # Akım gerçekte 0 ama sensör 1.0A okuyor varsayımı
    for _ in range(3600):  # 1 saat
        cc.step(current_a=1.0, dt_s=1.0)

    # 1 saatte 1 Ah kaymalı: %80 -> %70 olmalı
    assert pytest.approx(cc.soc, 0.02) == 0.70


def test_ekf_hatali_baslangictan_yakinsama():
    """EKF'nin yanlış başlangıç tahminini (%40) doğru değere (%80) hızla yaklaştırdığı test edilir."""
    ekf = BatteryEKFSoCEstimator(initial_soc_guess=0.40, capacity_ah=75.0)

    # Gerçek hücre: %80 SoC, 0A akım, Dinlenme voltajı = OCV(%80)
    ocv_true, _ = ekf._compute_ocv_and_derivative(0.80)

    for _ in range(200):
        out = ekf.step(current_a=0.0, measured_terminal_v=ocv_true, dt_s=0.1)

    # 20 saniye sonra EKF %80'e çok yaklaşmalıdır
    assert pytest.approx(out["estimated_soc"], 0.03) == 0.80
    assert out["soc_uncertainty_std"] < 0.05  # Kovaryans küçülmüştür


def test_ekf_ocv_ve_turev_dogrulugu():
    """Analitik OCV ve d(OCV)/d(SoC) türev fonksiyonunun pozitif ve sürekli olduğu test edilir."""
    ekf = BatteryEKFSoCEstimator()
    ocv, docv = ekf._compute_ocv_and_derivative(0.50)

    assert 3.50 < ocv < 3.90
    assert docv > 0.0  # OCV eğrisi monotondur


def test_ekf_dinamik_akım_takibi():
    """Dinamik şarj/deşarj darbeleri altında EKF'nin stabil kaldığı test edilir."""
    ekf = BatteryEKFSoCEstimator(initial_soc_guess=0.80, capacity_ah=75.0)
    ocv_init, _ = ekf._compute_ocv_and_derivative(0.80)

    for i in range(100):
        i_cur = 50.0 if i % 2 == 0 else -30.0
        v_meas = ocv_init - (i_cur * 0.0015)
        out = ekf.step(current_a=i_cur, measured_terminal_v=v_meas, dt_s=0.1)

    assert 0.0 <= out["estimated_soc"] <= 1.0
