"""
Day 350: Beyond Visual Range (BVR) Air Combat Multi-Agent Reinforcement Learning (MARL)
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

from src.bvr_air_combat_motoru import (
    BVRFighterAgent,
    ActiveRadarMissile,
    MARLTacticalPolicy,
    BVRAirCombatArena,
)
from src.bvr_profilleyici import BVRProfilleyici


def test_bvr_fighter_creation_and_radar_lock():
    """
    Savaş Uçağı Radar Koni Kilidi Testi.
    """
    f = BVRFighterAgent("F1", "BLUE", np.array([0.0, 0.0]), np.deg2rad(0.0))
    tgt_in_front = np.array([20.0, 5.0]) # 14 derece -> Kilitlenmeli
    tgt_behind = np.array([-20.0, 0.0]) # 180 derece -> Kilitlenemez
    
    assert f.has_radar_lock(tgt_in_front) is True
    assert f.has_radar_lock(tgt_behind) is False


def test_active_radar_missile_step_and_pitbull():
    """
    ARH Füzesi Pitbull Fazı ve Güdüm Testi.
    """
    msl = ActiveRadarMissile("M1", "F1", "T1", np.array([0.0, 0.0]), 0.0)
    tgt_pos = np.array([10.0, 0.0]) # 10 km (< 15 km Pitbull)
    
    msl.step(0.1, tgt_pos, shooter_has_radar_lock=False)
    assert msl.is_pitbull is True
    assert msl.is_active is True


def test_marl_tactical_policy_decisions():
    """
    MARL Taktik Karar Politikası Testi.
    """
    b = BVRFighterAgent("B1", "BLUE", np.array([0.0, 0.0]), 0.0)
    r = BVRFighterAgent("R1", "RED", np.array([40.0, 0.0]), np.pi)
    
    msl = MARLTacticalPolicy.decide_action(b, r, [], [], 0.1)
    assert msl is not None
    assert b.missile_ammo == 1 # 1 füze ateşlendi


def test_bvr_profiler_metrics():
    """
    BVR Profilleyici Metrik Testi.
    """
    metrics = BVRProfilleyici.profille(
        blue_alive=2,
        red_alive=0,
        tactical_states=["INTERCEPT", "CRANK", "DRAG_PUMP"]
    )
    
    assert metrics["blue_survival"] == 100.0
    assert metrics["red_destruction"] == 100.0
    assert metrics["air_dominance_score"] > 95.0
