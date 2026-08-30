"""
Day 347: Decentralized Drone Swarm Flocking with Graph Neural Networks (GNN)
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

from src.drone_flocking_gnn_motoru import (
    DroneSwarmDynamicGraph,
    GNNFlockingController,
    DecentralizedSwarmSimulator,
)
from src.flocking_profilleyici import FlockingProfilleyici


def test_drone_swarm_dynamic_graph():
    """
    Dinamik İHA Graf Yapısı Testi.
    """
    graph = DroneSwarmDynamicGraph(num_drones=4, comm_radius_m=50.0)
    pos = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [100.0, 0.0, 0.0], # Menzil dışı
        [0.0, 20.0, 0.0]
    ])
    
    adj = graph.build_adjacency(pos)
    assert adj.shape == (4, 4)
    assert adj[0, 1] == 1.0 # 10 m menzilde
    assert adj[0, 2] == 0.0 # 100 m menzil dışında


def test_gnn_flocking_controller_separation():
    """
    GNN Flocking Kontrolcüsü Ayrılma (Separation) Kuvveti Testi.
    """
    ctrl = GNNFlockingController(d_desired_m=20.0)
    pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]) # 5m yakın
    vel = np.zeros((2, 3))
    adj = np.array([[0, 1], [1, 0]])
    tgt = np.array([50.0, 50.0, 50.0])
    
    u = ctrl.compute_gnn_control(pos, vel, adj, tgt)
    assert u.shape == (2, 3)
    # Drone 0, Drone 1'den sola (-X) kaçmalı
    assert u[0, 0] < u[1, 0]


def test_decentralized_swarm_step():
    """
    Merkeziyetsiz Sürü Simülatörü Adım Testi.
    """
    sim = DecentralizedSwarmSimulator(num_drones=6, dt=0.05)
    tgt = np.array([50.0, 50.0, 50.0])
    
    res = sim.step_simulation(tgt)
    assert "positions" in res
    assert "min_distance_m" in res
    assert not np.isnan(res["positions"]).any()


def test_flocking_profiler_metrics():
    """
    Flocking Profilleyici Metrik Testi.
    """
    metrics = FlockingProfilleyici.profille(
        min_inter_drone_dist_m=5.5,
        final_velocity_var=0.2,
        final_goal_dist_m=8.0
    )
    
    assert metrics["safety_score"] == 100.0
    assert metrics["alignment_score"] == 100.0
    assert metrics["goal_reach_score"] == 100.0
    assert metrics["swarm_flocking_readiness"] > 95.0
