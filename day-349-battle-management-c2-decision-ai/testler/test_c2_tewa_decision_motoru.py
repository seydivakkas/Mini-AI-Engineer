"""
Day 349: Battle Management Language (BML) & C2 Decision Support AI (TEWA)
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

from src.c2_tewa_decision_motoru import (
    BattlefieldThreat,
    DefenseAsset,
    TEWAOptimizer,
    BMLOrderGenerator,
    BattleManagementEngine,
)
from src.c2_profilleyici import C2Profilleyici


def test_battlefield_threat_and_asset_creation():
    """
    Tehdit ve Savunma Unsuru Veri Modelleri Testi.
    """
    th = BattlefieldThreat("T1", "CRUISE_MISSILE", np.array([10.0, 10.0, 1.0]), np.array([-0.5, 0.0, 0.0]), 90.0)
    ast = DefenseAsset("A1", "SAM_HISAR_O", np.array([0.0, 0.0, 0.0]), 50.0, 2, 0.9)
    
    assert th.threat_value == 90.0
    assert ast.ammo_remaining == 2


def test_tewa_optimizer_assignment():
    """
    TEWA Silah Tahsis Optimizatörü Testi.
    """
    threats = [
        BattlefieldThreat("T1", "CRUISE_MISSILE", np.array([20.0, 0.0, 1.0]), np.zeros(3), 95.0),
        BattlefieldThreat("T2", "FIGHTER_JET", np.array([150.0, 0.0, 10.0]), np.zeros(3), 80.0) # Menzil dışı
    ]
    assets = [
        DefenseAsset("A1", "SAM_HISAR_O", np.array([0.0, 0.0, 0.0]), max_range_km=50.0, ammo_remaining=1, base_pk=0.9)
    ]
    
    opt = TEWAOptimizer()
    assignments = opt.solve_assignment(threats, assets)
    
    assert len(assignments) == 1
    assert assignments[0]["threat_id"] == "T1"
    assert assignments[0]["assigned_asset_id"] == "A1"
    assert assignments[0]["expected_pk"] > 0.5


def test_bml_order_generator_structure():
    """
    NATO C-BML Emir Yapısı Testi (5W Kuralı).
    """
    asgn = {
        "threat_id": "T1",
        "threat_type": "CRUISE_MISSILE",
        "assigned_asset_id": "A1",
        "assigned_asset_type": "SAM_HISAR_O",
        "expected_pk": 0.85,
        "target_distance_km": 15.0
    }
    th = BattlefieldThreat("T1", "CRUISE_MISSILE", np.array([10.0, 10.0, 1.0]), np.zeros(3), 90.0)
    
    order = BMLOrderGenerator.generate_bml_order(asgn, th)
    assert "WHO" in order
    assert "WHAT" in order
    assert "WHERE" in order
    assert "WHEN" in order
    assert "WHY" in order
    assert order["WHO"] == "A1"


def test_c2_profiler_metrics():
    """
    C2 Karar Destek Profilleyici Testi.
    """
    asgns = [{"expected_pk": 0.88}, {"expected_pk": 0.92}]
    metrics = C2Profilleyici.profille(num_threats=2, assignments=asgns, decision_time_ms=0.35)
    
    assert metrics["threat_coverage_score"] == 100.0
    assert metrics["tewa_efficiency_score"] == 90.0
    assert metrics["c2_readiness_score"] > 95.0
