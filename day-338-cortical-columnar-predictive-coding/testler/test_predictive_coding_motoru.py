"""
Day 338: Cortical Column Architecture & Hierarchical Predictive Coding
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

from src.predictive_coding_motoru import (
    CorticalColumnLayer,
    HierarchicalCorticalNetwork,
    FreeEnergyMinimizer,
)
from src.cortical_profilleyici import CorticalProfilleyici


def test_cortical_column_layer_forward():
    """
    Kortikal Kolon Katman İleri Tahmin ve Hata Testi.
    """
    layer = CorticalColumnLayer(in_dim=16, state_dim=8)
    y_in = np.ones(16)
    
    err = layer.compute_error(y_in)
    assert len(err) == 16
    assert layer.W.shape == (16, 8)


def test_hierarchical_cortical_network_inference():
    """
    Hiyerarşik Öngörücü Kodlama Çıkarım ve Enerji Düşüş Testi.
    """
    net = HierarchicalCorticalNetwork(layer_dims=[32, 16, 8])
    sensory = np.random.randn(32)
    
    res = net.infer_and_reconstruct(sensory, n_steps=20)
    assert len(res["reconstructed_input"]) == 32
    assert res["final_free_energy"] < res["free_energy_history"][0]


def test_free_energy_minimizer_reduction():
    """
    Serbest Enerji Düşüş Oranı Testi.
    """
    red = FreeEnergyMinimizer.calculate_free_energy_reduction(initial_energy=10.0, final_energy=1.0)
    assert abs(red - 90.0) < 1e-4


def test_cortical_profiler_metrics():
    """
    Kortikal Kolon Profilleyici Metrik Doğrulaması.
    """
    metrics = CorticalProfilleyici.profille(energy_reduction_pct=95.0, reconstruction_mse=0.01, snr_gain_db=15.0)
    
    assert metrics["energy_reduction_score"] == 95.0
    assert metrics["cortical_readiness_score"] > 90.0
