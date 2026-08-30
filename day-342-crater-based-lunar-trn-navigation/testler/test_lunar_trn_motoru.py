"""
Day 342: Crater-Based Lunar Terrain Relative Navigation (TRN) for Precision Landing
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

from src.lunar_trn_motoru import (
    LunarCraterDatabase,
    OpticalCraterDetector,
    TerrainRelativeNavigator,
    HazardAvoidancePlanner,
)
from src.trn_profilleyici import TRNProfilleyici


def test_crater_database_structure():
    """
    Ay Krater Kataloğu Veritabanı Yapı Testi.
    """
    db = LunarCraterDatabase()
    craters = db.get_craters()
    
    assert craters.shape[0] >= 5
    assert craters.shape[1] == 4


def test_optical_crater_projection():
    """
    Optik Kamera Krater İzdüşüm Testi.
    """
    db = LunarCraterDatabase()
    detector = OpticalCraterDetector(focal_length_px=1000.0)
    lander_pos = np.array([0.0, 0.0, 10.0])
    
    detected = detector.project_craters(lander_pos, db.get_craters())
    assert len(detected) > 0
    assert "u" in detected[0]
    assert "radius_px" in detected[0]


def test_trn_pose_estimation():
    """
    TRN 3D Konum Kestirim Doğruluğu Testi.
    """
    db = LunarCraterDatabase()
    detector = OpticalCraterDetector(focal_length_px=1000.0, noise_px=0.0)
    navigator = TerrainRelativeNavigator(database=db)
    
    true_pos = np.array([1.0, -1.0, 8.0])
    detected = detector.project_craters(true_pos, db.get_craters())
    res = navigator.estimate_lander_pose(detected, focal_length=1000.0)
    
    assert res["success"] is True
    err_km = np.linalg.norm(true_pos - res["estimated_pos"])
    assert err_km < 0.01  # 10 metreden az hata


def test_hazard_avoidance_planner():
    """
    HDA Tehlike Tespiti ve Güvenli Sapma Planlama Testi.
    """
    db = LunarCraterDatabase()
    hda = HazardAvoidancePlanner(safety_margin_km=0.8)
    
    # Merkez kraterin tam üzerine iniş hedefi (Tehlikeli)
    dangerous_target = np.array([0.0, 0.0, 0.0])
    hda_res = hda.evaluate_landing_safety(dangerous_target, db.get_craters())
    
    assert hda_res["is_safe"] is False
    assert hda_res["divert_distance_m"] > 500.0
