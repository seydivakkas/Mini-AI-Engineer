"""
Day 355: Liquid Rocket Engine Health Monitoring & Time-Series Transformer Anomaly Detection
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

from src.rocket_health_transformer_motoru import (
    RocketEngineTelemetrySimulator,
    RocketHealthTransformerEngine,
    EngineAnomalyDetector,
    AutonomousAbortController,
)
from src.rocket_profilleyici import RocketProfilleyici


def test_rocket_engine_telemetry_simulator():
    """
    Roket Telemetri Simülatörü Boyut ve Değer Testi.
    """
    sim = RocketEngineTelemetrySimulator(seq_len=200)
    nom = sim.generate_nominal_telemetry()
    assert nom.shape == (200, 4)
    
    corrupted = sim.inject_turbopump_bearing_anomaly(nom, start_step=100)
    assert corrupted.shape == (200, 4)
    assert corrupted[-1, 3] > nom[-1, 3] # Titreşim artışı


def test_rocket_health_transformer_attention():
    """
    Transformer Self-Attention Kestirim Testi.
    """
    transformer = RocketHealthTransformerEngine()
    X = np.random.normal(100.0, 5.0, (50, 4))
    recon = transformer.compute_self_attention(X)
    assert recon.shape == (50, 4)
    assert not np.isnan(recon).any()


def test_engine_anomaly_detector_scores():
    """
    Anomali Skoru ve Eşik Tespiti Testi.
    """
    detector = EngineAnomalyDetector(threshold=15.0)
    raw = np.array([[160.0, 42.0, 850.0, 12.0], [160.0, 30.0, 950.0, 45.0]])
    pred = np.array([[160.0, 42.0, 850.0, 12.0], [160.0, 42.0, 850.0, 12.0]])
    scores = detector.compute_anomaly_scores(raw, pred)
    
    assert scores[0] < 1.0 # Nominal
    assert scores[1] > 50.0 # Anomali


def test_autonomous_abort_controller():
    """
    Otonom Acil Kapatma Kontrolcüsü Testi.
    """
    controller = AutonomousAbortController(abort_threshold=20.0, consecutive_triggers=3)
    scores = np.array([5.0] * 50 + [25.0] * 20)
    res = controller.evaluate_abort(scores)
    
    assert res["abort_triggered"] is True
    assert res["abort_step"] == 52
    assert res["safe_shutdown_achieved"] is True
