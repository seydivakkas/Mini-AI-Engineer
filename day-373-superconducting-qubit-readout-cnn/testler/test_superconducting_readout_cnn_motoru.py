"""
Day 373: Superconducting Qubit State Readout via Deep 1D-CNN
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

from src.superconducting_readout_cnn_motoru import (
    DispersiveReadoutSimulator,
    QubitReadoutCNN,
    QubitReadoutBenchmark,
)
from src.readout_profilleyici import ReadoutProfilleyici


def test_dispersive_readout_simulator_trace_shape():
    """
    Mikrodalga Zaman Serisi Simülatörü Şekil Testi.
    """
    sim = DispersiveReadoutSimulator(time_steps=64)
    traces, labels = sim.generate_traces(num_samples=50)
    
    assert traces.shape == (50, 2, 64)
    assert len(labels) == 50


def test_qubit_readout_cnn_forward():
    """
    1D-CNN Kubit Sınıflandırıcı İleri Geçiş Testi.
    """
    cnn = QubitReadoutCNN()
    dummy_input = np.random.randn(10, 2, 64).astype(np.float32)
    probs = cnn.forward(dummy_input)
    
    assert probs.shape == (10, 3)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_qubit_readout_benchmark():
    """
    Kubit Okuma Sadakati (Fidelity) Kıyaslama Testi.
    """
    benchmark = QubitReadoutBenchmark()
    res = benchmark.run_benchmark()
    
    assert res["cnn_fidelity"] >= 98.5
    assert res["discrimination_time_ns"] <= 200.0


def test_readout_profiler_metrics():
    """
    Kubit Okuma Profilleyici Metrik Testi.
    """
    mock_res = {
        "classical_fidelity": 91.0,
        "cnn_fidelity": 99.4,
        "fidelity_gain": 8.4,
        "discrimination_time_ns": 120.0
    }
    metrics = ReadoutProfilleyici.profille(mock_res)
    assert metrics["fidelity_score"] >= 99.0
    assert metrics["readout_readiness_score"] >= 98.0
