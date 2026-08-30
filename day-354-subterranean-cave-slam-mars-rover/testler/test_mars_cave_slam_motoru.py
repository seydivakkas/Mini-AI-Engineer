"""
Day 354: Subterranean Lava Tube Exploration & GPS-Denied 3D Graph SLAM for Mars Rovers
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

from src.mars_cave_slam_motoru import (
    MartianLavaTubeCave,
    MarsRover3DGraphSLAM,
    SubterraneanExplorationEngine,
)
from src.cave_profilleyici import CaveProfilleyici


def test_martian_lava_tube_cave_points():
    """
    Mars Mağara Simülatörü Nokta Bulutu Testi.
    """
    cave = MartianLavaTubeCave(num_points=500)
    assert len(cave.cave_points) == 500
    assert len(cave.centerline) == 200
    
    scan = cave.sample_lidar_scan(np.array([0.0, 0.0, -20.0]))
    assert len(scan) > 0


def test_mars_rover_3d_graph_slam_loop_closure():
    """
    3D Graph SLAM Döngü Kapatma Testi.
    """
    slam = MarsRover3DGraphSLAM()
    # 100 adımlı çember yörünge
    t = np.linspace(0, 2*np.pi, 100)
    for i in range(100):
        pos = np.array([20.0 * np.cos(t[i]), 20.0 * np.sin(t[i]), 0.0]) + np.random.normal(0, 0.2, 3)
        slam.add_odometry_pose(pos)
        slam.detect_loop_closure(i)

    assert len(slam.loop_edges) > 0
    opt_poses = slam.optimize_pose_graph()
    assert len(opt_poses) == 100


def test_subterranean_exploration_engine_rmse():
    """
    Mars Keşif Motoru RMSE Azaltma Testi.
    """
    engine = SubterraneanExplorationEngine()
    res = engine.run_exploration()
    
    assert res["drift_rmse_m"] > res["slam_rmse_m"]
    assert res["slam_rmse_m"] < 3.0


def test_cave_profiler_metrics():
    """
    Mağara SLAM Profilleyici Testi.
    """
    metrics = CaveProfilleyici.profille(
        drift_rmse_m=5.0,
        slam_rmse_m=0.5,
        loop_count=10
    )
    assert metrics["loop_closure_score"] == 100.0
    assert metrics["drift_reduction_score"] == 90.0
    assert metrics["cave_slam_readiness"] > 90.0
