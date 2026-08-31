"""
Tesla Batarya SoH ve İç Direnç İzleyici Birim Testleri (PyTest)
==============================================================
Bu test paketi; SoH kapasite/direnç hesaplayıcılarını, RLS çevrimiçi parametre
tahminini ve döngüsel yaşlanma simülasyonunu test eder.

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

from src.tesla_soh_ve_ic_direnc_izleyici import (
    calculate_soh_capacity,
    calculate_soh_resistance,
    RecursiveLeastSquaresR0,
    BatteryCycleAgingSimulator
)


def test_soh_kapasite_hesabi():
    """Kapasite oranına göre SoH'ın doğru hesaplandığı test edilir."""
    assert calculate_soh_capacity(75.0, 75.0) == 100.0
    assert calculate_soh_capacity(60.0, 75.0) == 80.0
    assert calculate_soh_capacity(45.0, 75.0) == 60.0


def test_soh_direnc_hesabi():
    """İç direnç artışına göre direnç SoH'ının doğru hesaplandığı test edilir."""
    fresh_r0 = 0.0015
    assert pytest.approx(calculate_soh_resistance(0.0015, fresh_r0_ohm=fresh_r0)) == 100.0
    # 2 katına çıkınca %0 SoH olmalıdır (EOL)
    assert pytest.approx(calculate_soh_resistance(0.0030, fresh_r0_ohm=fresh_r0)) == 0.0
    # 1.5 katında %50 SoH olmalıdır
    assert pytest.approx(calculate_soh_resistance(0.00225, fresh_r0_ohm=fresh_r0)) == 50.0


def test_rls_parametre_kestirimi():
    """RLS algoritmasının gerçek iç dirence (2.0 mOhm) yakınsadığı test edilir."""
    rls = RecursiveLeastSquaresR0(initial_r0_guess=0.0010, lambda_forgetting=0.99)
    true_r0 = 0.0020

    for i in range(200):
        d_i = float(50.0 + 10.0 * np.sin(i * 0.1))
        d_v = d_i * true_r0
        tahmin = rls.update(d_i, d_v)

    assert pytest.approx(tahmin, 0.05) == true_r0


def test_dongusel_yaslanma_simulasyonu():
    """1000 döngü yaşlandırma sonrası kapasitenin azalıp direncin arttığı test edilir."""
    sim = BatteryCycleAgingSimulator(fresh_capacity_ah=75.0, fresh_r0_ohm=0.0015)
    sim.step_cycles(cycle_count=1000, temp_c=35.0, dod_depth_of_discharge=0.80)
    durum = sim.get_health_status()

    assert durum["capacity_ah"] < 75.0
    assert durum["r0_ohm"] > 0.0015
    assert durum["soh_capacity_pct"] < 100.0
