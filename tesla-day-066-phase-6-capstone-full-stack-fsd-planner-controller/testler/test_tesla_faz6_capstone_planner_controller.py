"""
Tesla Faz 6 Capstone Birim Testleri (PyTest)
============================================
Bu test paketi; Full-Stack FSD Planlayıcı ve Kontrolcü motorunun Quintic yörünge
sentezini, MPC/Stanley kapalı çevrim takibini ve ASIL-D / Çift Düğüm onayını test eder.

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

from src.tesla_faz6_capstone_planner_controller import TeslaFullStackFSDPlannerController


def test_quintic_serit_degistirme_yorungesi():
    """Quintic polinomunun 3.5m şerit değişimini ve <3.5 m/s^3 jerk konfor limitini sağladığı test edilir."""
    engine = TeslaFullStackFSDPlannerController(target_lane_width_m=3.5, cruise_speed_mps=25.0, time_horizon_s=4.0)
    traj = engine.plan_quintic_trajectory(steps=50)

    assert np.isclose(traj["lateral_d"][-1], 3.5, atol=1e-3)
    assert np.isclose(traj["lateral_vel"][-1], 0.0, atol=1e-3)
    assert np.max(np.abs(traj["lateral_jerk"])) <= 3.50


def test_mpc_stanley_kapali_cevrim_takip():
    """MPC/Stanley takipçisinin yanal hatayı <8 cm ve açı hatasını <1.5 derece tuttuğu test edilir."""
    engine = TeslaFullStackFSDPlannerController()
    traj = engine.plan_quintic_trajectory(steps=50)
    tracking = engine.compute_mpc_stanley_tracking(traj)

    assert tracking["is_tracking_accurate"] is True
    assert tracking["final_lateral_error_m"] < 0.08
    assert tracking["final_yaw_error_deg"] < 1.50


def test_full_stack_fsd_capstone_pipeline():
    """Tüm Faz 6 planlama, kontrol, ASIL-D ve çift düğüm arabulucu zincirinin başarıyla çalıştığı test edilir."""
    engine = TeslaFullStackFSDPlannerController()
    res = engine.run_full_fsd_pipeline()

    assert res["success"] is True
    assert res["asil_d_verified"] is True
    assert res["arbiter_consensus"] is True
    assert res["aeb_status"] == "NORMAL (GÜVENLİ TAKİP)"
