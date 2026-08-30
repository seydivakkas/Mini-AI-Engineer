"""
Day 341: Spacecraft Autonomous GNC (Guidance, Navigation & Control) under Zero-GNSS
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Birim Test Paketi (PyTest Suite)
"""

import sys
import os
import pytest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.spacecraft_gnc_motoru import (
    OpticalStarTracker,
    OrbitalEKFNavigator,
    AutonomousGNCController,
)
from src.gnc_profilleyici import GNCProfilleyici


def test_optical_star_tracker_triad():
    """
    Optik Yıldız Takipçisi TRIAD Yönelim Kestirim Testi.
    """
    tracker = OpticalStarTracker(noise_std=0.001)
    R_true = np.eye(3)
    v1_b, v2_b = tracker.measure_body_vectors(R_true)
    R_est = tracker.triad_attitude_estimation(v1_b, v2_b)
    
    err = np.linalg.norm(R_true - R_est)
    assert err < 0.01


def test_orbital_ekf_j2_acceleration():
    """
    İki Cisim + J2 Yerçekimi İvmesi Hesabı Testi.
    """
    r = np.array([7000.0, 0.0, 100.0])
    ekf = OrbitalEKFNavigator(initial_state=np.array([7000.0, 0.0, 100.0, 0.0, 7.5, 0.0]))
    a_grav = ekf.gravitational_acceleration(r)
    
    assert len(a_grav) == 3
    assert np.linalg.norm(a_grav) > 0.0


def test_orbital_ekf_propagation_update():
    """
    EKF Yörünge Durum Öteleme ve Ölçüm Güncelleme Testi.
    """
    state_0 = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    ekf = OrbitalEKFNavigator(initial_state=state_0, dt=1.0)
    
    ekf.propagate_state()
    assert np.linalg.norm(ekf.state[:3] - state_0[:3]) > 0.0
    
    z_meas = np.array([7001.0, 0.0, 0.0])
    ekf.measurement_update(z_meas)
    assert ekf.state[0] > 7000.0


def test_gnc_profiler_metrics():
    """
    GNC Profilleyici Metrik Doğrulaması.
    """
    metrics = GNCProfilleyici.profille(mean_pos_error_m=1.2, mean_attitude_error_deg=0.02)
    
    assert metrics["attitude_score"] == 100.0
    assert metrics["orbit_accuracy_score"] == 100.0
    assert metrics["gnc_readiness_score"] > 95.0
