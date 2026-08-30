"""
Day 348: Degraded Visual Environment (DVE) Sensor Fusion (LiDAR + Radar + FLIR)
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

from src.dve_sensor_fusion_motoru import (
    DVESensorSimulator,
    AdaptiveDVEFusionEngine,
    ObstacleGridMapper,
)
from src.dve_profilleyici import DVEProfilleyici


def test_dve_sensor_simulator():
    """
    DVE Çoklu-Sensör Simülatörü Testi.
    """
    obs = np.array([[10.0, 10.0, 2.0], [-10.0, 15.0, 3.0]])
    sim = DVESensorSimulator(true_obstacles=obs)
    data = sim.sample_sensors(degradation_gamma=0.8)
    
    assert "lidar_meas" in data
    assert "radar_meas" in data
    assert "flir_meas" in data
    assert len(data["radar_meas"]) == 2


def test_adaptive_dve_fusion_engine_precision():
    """
    Adaptif DVE Füzyon Motoru Doğruluk Testi.
    """
    np.random.seed(42)
    obs = np.array([[15.0, 20.0, 1.0], [-15.0, 20.0, 2.0]])
    sim = DVESensorSimulator(true_obstacles=obs)
    engine = AdaptiveDVEFusionEngine()
    
    data = sim.sample_sensors(degradation_gamma=0.8)
    fused_pos, fused_vars = engine.fuse_measurements(data)
    
    assert fused_pos.shape == (2, 3)
    fused_err = np.linalg.norm(fused_pos - obs, axis=-1)
    # Füzyon hatası 0.85 metrenin altında kalmalı (3D norm)
    assert np.all(fused_err < 0.85)


def test_obstacle_grid_mapper():
    """
    Emniyetli İniş Bölgesi Kontrolü Testi.
    """
    mapper = ObstacleGridMapper(safe_radius_m=10.0)
    safe_obs = np.array([[20.0, 20.0, 1.0], [-25.0, 25.0, 2.0]]) # 10 metreden uzak
    unsafe_obs = np.array([[3.0, 4.0, 1.0]]) # 5 metrede engel

    assert mapper.evaluate_safe_landing_zone(np.array([0, 0, 0]), safe_obs) is True
    assert mapper.evaluate_safe_landing_zone(np.array([0, 0, 0]), unsafe_obs) is False


def test_dve_profiler_metrics():
    """
    DVE Profilleyici Testi.
    """
    errors = {
        "lidar_rmse": 0.85,
        "radar_rmse": 0.45,
        "flir_rmse": 0.38,
        "fused_rmse": 0.18
    }
    metrics = DVEProfilleyici.profille(errors_dict=errors, safe_landing=True)
    assert metrics["fusion_accuracy_score"] > 80.0
    assert metrics["dve_safety_score"] == 100.0
