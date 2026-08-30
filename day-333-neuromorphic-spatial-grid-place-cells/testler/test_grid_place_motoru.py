"""
Day 333: Neuromorphic Spatial Navigation & Grid/Place Cells
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

from src.grid_place_motoru import (
    GridCellModule,
    PlaceCellNetwork,
    NeuromorphicSpatialNavigator,
)
from src.grid_place_profilleyici import GridPlaceProfilleyici


def test_grid_cell_module_hexagonal_firing():
    """
    Entorhinal Grid Hücresi Hekzagonal Ateşleme Oranı Testi.
    """
    grid_mod = GridCellModule(spatial_scale=1.5)
    rate = grid_mod.compute_firing_rate(np.array([0.0, 0.0], dtype=np.float32))
    
    assert 0.0 <= rate <= 1.0


def test_place_cell_network_decoding():
    """
    Hipokampal Konum Hücreleri Popülasyon Kod Çözümü Testi.
    """
    place_net = PlaceCellNetwork(grid_size=5, env_bounds=4.0)
    target_pos = np.array([0.5, -0.5], dtype=np.float32)
    
    rates = place_net.compute_place_rates(target_pos)
    decoded_pos = place_net.decode_position(rates)
    
    error = np.linalg.norm(target_pos - decoded_pos)
    assert error < 0.3  # Düşük kod çözme hatası


def test_neuromorphic_spatial_navigator_step():
    """
    Yol Entegrasyonu Navigasyon Adım Testi.
    """
    navigator = NeuromorphicSpatialNavigator(initial_position=(0.0, 0.0))
    vel = np.array([1.0, 0.5], dtype=np.float32)
    
    res = navigator.update_navigation_step(vel, dt=0.1)
    assert "true_pos" in res
    assert "decoded_pos" in res
    assert "error_meters" in res
    assert res["error_meters"] >= 0.0


def test_grid_place_profiler_metrics():
    """
    Grid/Place Profilleyici Metrik Doğrulaması.
    """
    metrics = GridPlaceProfilleyici.profille(
        mean_error_meters=0.12,
        hexagonal_symmetry_score=98.0
    )
    
    assert metrics["decoding_precision_score"] > 80.0
    assert metrics["navigation_readiness_score"] > 85.0
