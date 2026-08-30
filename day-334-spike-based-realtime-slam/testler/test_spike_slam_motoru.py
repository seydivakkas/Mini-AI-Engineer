"""
Day 334: Microsecond Latency Spike-based Neuromorphic SLAM
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

from src.spike_slam_motoru import (
    DVSEventStreamSimulator,
    SpikeScanMatcher,
    NeuromorphicOccupancyGridSLAM,
)
from src.spike_slam_profilleyici import SpikeSlamProfilleyici


def test_dvs_event_stream_simulator():
    """
    DVS Olay Akış Simülatörü Spike Üretim Testi.
    """
    sim = DVSEventStreamSimulator(map_size=50)
    events = sim.generate_event_batch(agent_pos=np.array([25.0, 25.0]), dt_us=1000)
    
    assert isinstance(events, list)
    if len(events) > 0:
        e = events[0]
        assert len(e) == 4  # (x, y, t_us, polarity)


def test_spike_scan_matcher():
    """
    Spike Tabanlı ICP Taraması Hizalama Testi.
    """
    pts1 = np.array([[10, 10], [10, 11], [11, 10]], dtype=np.float32)
    pts2 = np.array([[12, 12], [12, 13], [13, 12]], dtype=np.float32)
    
    dx, dy, d_th = SpikeScanMatcher.match_scans(pts1, pts2)
    assert abs(dx - 2.0) < 1e-3
    assert abs(dy - 2.0) < 1e-3


def test_neuromorphic_occupancy_grid_slam():
    """
    Mikrosaniye Gecikmeli SLAM İşlem ve Mikrosaniye Hız Testi.
    """
    slam = NeuromorphicOccupancyGridSLAM(map_size=50)
    events = [(15, 15, 100, 1), (15, 16, 101, 1)]
    
    res = slam.process_event_batch(events)
    assert "estimated_pose" in res
    assert "occupancy_prob" in res
    assert "latency_us" in res
    # Mikrosaniye gecikme < 1000 us (1 ms) olmalı
    assert res["latency_us"] < 1000.0


def test_spike_slam_profiler_metrics():
    """
    SLAM Profilleyici Metrik Doğrulaması.
    """
    metrics = SpikeSlamProfilleyici.profille(
        mean_pose_error=0.45,
        mean_latency_us=12.5,
        mapping_accuracy=96.0
    )
    
    assert metrics["latency_speed_score"] > 90.0
    assert metrics["slam_readiness_score"] > 85.0
