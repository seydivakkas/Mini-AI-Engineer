"""
Day 353: Active Space Debris Laser Ablation & Multi-Target Deorbiting Path Optimization
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

from src.space_debris_laser_motoru import (
    SpaceDebrisObject,
    LaserAblationImpulseEngine,
    MultiDebrisTSPPathOptimizer,
    ActiveDebrisRemovalMission,
)
from src.debris_profilleyici import DebrisProfilleyici


def test_space_debris_object_creation():
    """
    Uzay Çöpü Nesnesi Veri Modeli Testi.
    """
    d = SpaceDebrisObject("D1", 100.0, 750.0, 65.0, 90.0)
    assert d.mass_kg == 100.0
    assert d.altitude_km == 750.0


def test_laser_ablation_deorbit_calculation():
    """
    Lazer İtki ve Yörünge İndirme Testi.
    """
    engine = LaserAblationImpulseEngine()
    d = SpaceDebrisObject("D1", 150.0, 800.0, 70.0, 85.0)
    res = engine.calculate_deorbit_shots(d, target_perigee_km=180.0)
    
    assert res["successful_deorbit"] is True
    assert res["required_laser_shots"] > 0
    assert res["delta_v_required_ms"] > 100.0 # ~150-180 m/s


def test_multi_debris_tsp_path_optimizer():
    """
    Çoklu Enkaz Rota Optimizatörü Testi.
    """
    opt = MultiDebrisTSPPathOptimizer()
    debris_list = [
        SpaceDebrisObject("D1", 100.0, 800.0, 60.0, 95.0),
        SpaceDebrisObject("D2", 200.0, 810.0, 62.0, 70.0),
        SpaceDebrisObject("D3", 150.0, 790.0, 61.0, 80.0),
    ]
    route, cost = opt.optimize_visit_sequence(debris_list)
    assert len(route) == 3
    assert set(route) == {0, 1, 2}
    assert cost > 0.0


def test_debris_profiler_metrics():
    """
    Uzay Çöpü Profilleyici Metrik Testi.
    """
    mock_res = {
        "total_cleaned": 5,
        "total_transfer_dv_ms": 320.0
    }
    metrics = DebrisProfilleyici.profille(mock_res)
    assert metrics["deorbit_success_score"] == 100.0
    assert metrics["kessler_mitigation_score"] > 95.0
