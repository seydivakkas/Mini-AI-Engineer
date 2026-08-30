"""
Day 356: Autonomous Aerial Refueling (AAR) Vision-Based Docking Flight Controller
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

from src.aar_docking_vision_motoru import (
    TankerDrogueKinematicsSimulator,
    VisionBasedDrogueTracker,
    AARDockingFlightController,
    AutonomousAerialRefuelingMission,
)
from src.aar_profilleyici import AARProfilleyici


def test_tanker_drogue_kinematics():
    """
    Tanker Sepet Hareketi Kinematik Testi.
    """
    tanker = TankerDrogueKinematicsSimulator(baseline_dist_m=25.0)
    p = tanker.get_drogue_position(5.0)
    assert len(p) == 3
    assert p[0] == pytest.approx(25.0, abs=1.0)


def test_vision_based_drogue_tracker():
    """
    Görü Tabanlı Sepet Takipçisi Testi.
    """
    tracker = VisionBasedDrogueTracker()
    drogue_p = np.array([20.0, 0.5, -2.0])
    uav_p = np.array([0.0, 0.0, 0.0])
    
    est = tracker.track_drogue(drogue_p, uav_p)
    assert len(est) == 3
    assert est[0] > 1.0


def test_aar_docking_flight_controller():
    """
    AAR Kenetlenme Uçuş Kontrolcüsü Testi.
    """
    controller = AARDockingFlightController()
    rel_est = np.array([10.0, 0.8, -0.5])
    cmd = controller.compute_control_acceleration(rel_est, approach_speed_ms=0.5, dt=0.05)
    
    assert len(cmd) == 3
    assert cmd[0] == 0.5 # İleri yaklaşma ivmesi
    assert cmd[1] > 0.0 # Yanal düzeltme


def test_aar_profiler_metrics():
    """
    AAR Profilleyici Metrik Testi.
    """
    mock_res = {
        "docked": True,
        "final_lateral_error_cm": 4.5
    }
    metrics = AARProfilleyici.profille(mock_res)
    assert metrics["docked"] is True
    assert metrics["vision_tracking_score"] == 100.0
    assert metrics["aar_mission_success_score"] > 90.0
