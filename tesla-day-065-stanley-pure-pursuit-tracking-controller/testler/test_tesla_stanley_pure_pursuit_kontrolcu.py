"""
Tesla Stanley ve Pure Pursuit Birim Testleri (PyTest)
======================================================
Bu test paketi; Stanley geometrik takip kontrol formülasyonunu,
Pure Pursuit bakış açısı hesabını ve kapalı çevrim takip yakınsamasını test eder.

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

from src.tesla_stanley_pure_pursuit_kontrolcu import TeslaStanleyTracker, TeslaPurePursuitTracker, TeslaTrackingBenchmark


def test_stanley_kontrol_formulu():
    """Stanley kontrolcüsünün theta_e + atan(k*e / (v+eps)) değerini doğru hesapladığı test edilir."""
    tracker = TeslaStanleyTracker(gain_k=0.50, softening_eps=0.10)

    # heading_error = 0.04 rad, cross_track_error = 0.30 m, v = 15.0 m/s
    steer = tracker.compute_steering(heading_error_rad=0.04, cross_track_error_m=0.30, speed_mps=15.0)

    beklenen_atan = np.arctan2(0.50 * 0.30, 15.0 + 0.10)
    beklenen_steer = 0.04 + beklenen_atan
    assert np.isclose(steer, beklenen_steer)


def test_pure_pursuit_kontrol_formulu():
    """Pure pursuit kontrolcüsünün atan(2*L*sin(alpha)/L_d) değerini doğru hesapladığı test edilir."""
    tracker = TeslaPurePursuitTracker(wheelbase_m=2.875, lookahead_gain=0.80, min_lookahead_m=3.0)

    # alpha = 0.05 rad, speed = 15.0 m/s -> L_d = 0.8 * 15.0 = 12.0m
    steer = tracker.compute_steering(alpha_rad=0.05, speed_mps=15.0)

    beklenen_steer = np.arctan2(2.0 * 2.875 * np.sin(0.05), 12.0)
    assert np.isclose(steer, beklenen_steer)


def test_stanley_kapali_cevrim_yakinsama():
    """50 adım sonunda takip hatasının 5 cm (0.05 m) altına indiği test edilir."""
    sim = TeslaTrackingBenchmark()
    res = sim.run_tracking_simulation(steps=50, speed_mps=15.0)

    assert res["is_converged"] is True
    assert res["final_lateral_error_m"] < 0.05
