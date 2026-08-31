"""
Tesla Frenet ve Quintic Polinom Birim Testleri (PyTest)
========================================================
Bu test paketi; Quintic polinom sınır koşullarını, yanal hız/ivme/jerk
türevlerini ve konfor kısıtlarını test eder.

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

from src.tesla_frenet_ve_quintic_serit_degistirme import TeslaFrenetTrajectoryPlanner


def test_quintic_sinir_kosullari_eslesmesi():
    """Başlangıçta d(0)=0, v(0)=0, a(0)=0 ve bitişte d(T)=3.5, v(T)=0, a(T)=0 olduğu test edilir."""
    planner = TeslaFrenetTrajectoryPlanner(target_lane_width_m=3.5, time_horizon_s=4.0)
    coeffs = planner.solve_quintic_polynomial(d0=0.0, v0=0.0, a0=0.0, d1=3.5, v1=0.0, a1=0.0, T=4.0)

    # t = 0 anı
    p_0 = planner.evaluate_trajectory_profiles(coeffs, np.array([0.0]))
    assert np.isclose(p_0["lateral_pos_d"][0], 0.0)
    assert np.isclose(p_0["lateral_vel_v"][0], 0.0)
    assert np.isclose(p_0["lateral_acc_a"][0], 0.0)

    # t = 4.0 anı
    p_T = planner.evaluate_trajectory_profiles(coeffs, np.array([4.0]))
    assert np.isclose(p_T["lateral_pos_d"][0], 3.5)
    assert np.isclose(p_T["lateral_vel_v"][0], 0.0, atol=1e-5)
    assert np.isclose(p_T["lateral_acc_a"][0], 0.0, atol=1e-5)


def test_yanal_konfor_ve_jerk_siniri():
    """Maksimum yanal jerk değerinin konfor sınırı <= 2.0 m/s^3 altında kaldığı test edilir."""
    planner = TeslaFrenetTrajectoryPlanner()
    res = planner.generate_frenet_lane_change(current_speed_mps=25.0)

    assert res["max_lateral_jerk"] <= 3.5
    assert res["max_lateral_acc"] <= 2.0
    assert res["is_comfortable"] is True


def test_boyuna_ve_yanal_profil_boyutlari():
    """50 adımlık zaman serisinde s ve d profillerinin tutarlı boyutta olduğu test edilir."""
    planner = TeslaFrenetTrajectoryPlanner()
    res = planner.generate_frenet_lane_change(steps=50)

    assert len(res["time_array"]) == 50
    assert len(res["longitudinal_s"]) == 50
    assert len(res["profiles"]["lateral_pos_d"]) == 50
