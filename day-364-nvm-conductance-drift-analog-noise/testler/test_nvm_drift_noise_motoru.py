"""
Day 364: Non-Volatile Memory (NVM) Conductance Drift & Analog Noise Compensation
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

from src.nvm_drift_noise_motoru import (
    PCMDriftNoiseSimulator,
    AdaptiveDriftCalibrator,
    DriftResilientInferenceEngine,
)
from src.drift_profilleyici import DriftProfilleyici


def test_pcm_drift_noise_simulator_decay():
    """
    PCM İletkenlik Kayması Güç Yasası Testi.
    """
    sim = PCMDriftNoiseSimulator(noise_ratio=0.0)
    g0 = np.ones((8, 8)) * 100e-6
    g_10s = sim.apply_drift_and_noise(g0, time_seconds=10.0)
    g_1000s = sim.apply_drift_and_noise(g0, time_seconds=1000.0)
    
    assert (np.mean(g_1000s) < np.mean(g_10s))
    assert (g_1000s > 0).all()


def test_adaptive_drift_calibrator_gain():
    """
    Adaptif Telafi Kazanç Hesaplama Testi.
    """
    cal = AdaptiveDriftCalibrator(ref_g0=100e-6)
    gain_nominal = cal.estimate_compensation_gain(100e-6)
    gain_drifted = cal.estimate_compensation_gain(50e-6)
    
    assert gain_nominal == pytest.approx(1.0, abs=0.05)
    assert gain_drifted == pytest.approx(2.0, abs=0.05)


def test_drift_resilient_inference_engine():
    """
    Çok Yıllı Çıkarım Doğruluk ve Telafi Testi.
    """
    engine = DriftResilientInferenceEngine(size=8)
    res = engine.run_multi_year_retention_benchmark()
    
    assert len(res["time_points"]) == 15
    assert res["final_comp_acc"] > res["final_uncomp_acc"]
    assert res["accuracy_recovery"] > 0.0


def test_drift_profiler_metrics():
    """
    NVM Drift Profilleyici Metrik Testi.
    """
    mock_res = {
        "final_uncomp_acc": 42.0,
        "final_comp_acc": 97.0,
        "accuracy_recovery": 55.0
    }
    metrics = DriftProfilleyici.profille(mock_res)
    assert metrics["final_comp_acc"] == 97.0
    assert metrics["accuracy_recovery"] == 55.0
    assert metrics["nvm_robustness_readiness"] > 97.0
