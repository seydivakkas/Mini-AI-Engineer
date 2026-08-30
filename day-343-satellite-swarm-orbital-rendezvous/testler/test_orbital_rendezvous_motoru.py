"""
Day 343: Satellite Swarm Orbital Rendezvous & Autonomous Collision Avoidance
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

from src.orbital_rendezvous_motoru import (
    ClohessyWiltshirePropagator,
    SwarmPotentialFieldCollisionAvoidance,
    AutonomousRendezvousController,
)
from src.rendezvous_profilleyici import RendezvousProfilleyici


def test_cw_propagator_step():
    """
    Clohessy-Wiltshire Bağıl Yörünge Öteleme Testi.
    """
    prop = ClohessyWiltshirePropagator(orbital_radius_km=7000.0)
    state_0 = np.array([0.1, 0.0, 0.0, 0.0, 0.001, 0.0])
    u_0 = np.zeros(3)
    
    state_1 = prop.step(state_0, u_0, dt=1.0)
    assert len(state_1) == 6
    assert state_1[1] != state_0[1] # Y yönünde ilerleme


def test_swarm_apf_repulsion():
    """
    APF Sürü İçi İtici Kuvvet Hesaplama Testi.
    """
    apf = SwarmPotentialFieldCollisionAvoidance(d_safe_m=40.0, k_rep=0.01)
    pos_a = np.array([0.0, 0.0, 0.0])
    pos_b = np.array([0.01, 0.0, 0.0]) # 10 metre yakın (40 m altında)
    
    f_rep = apf.compute_repulsion(pos_a, [pos_b])
    assert np.linalg.norm(f_rep) > 0.0
    assert f_rep[0] < 0.0 # B'den ters yöne itmeli


def test_autonomous_rendezvous_docking():
    """
    Otonom Kenetlenme Kontrolcüsü Kuvvet Hesaplama Testi.
    """
    prop = ClohessyWiltshirePropagator(orbital_radius_km=7000.0)
    apf = SwarmPotentialFieldCollisionAvoidance(d_safe_m=40.0)
    controller = AutonomousRendezvousController(propagator=prop, apf=apf)
    
    state = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 0.0])
    
    cmd = controller.compute_docking_control(state, target, other_deputy_positions=[])
    assert cmd["dist_to_docking_m"] == pytest.approx(50.0, abs=1.0)
    assert np.linalg.norm(cmd["u_thrust"]) > 0.0


def test_rendezvous_profiler_metrics():
    """
    Sürü Buluşma Profilleyici Doğrulama Testi.
    """
    metrics = RendezvousProfilleyici.profille(
        final_docking_dist_m=0.3,
        min_inter_sat_dist_m=35.0,
        collision_detected=False
    )
    
    assert metrics["collision_avoidance_score"] == 100.0
    assert metrics["docking_accuracy_score"] == 100.0
    assert metrics["swarm_rendezvous_readiness"] > 95.0
