"""
Day 340: Neuromorphic Bio-Cognitive Co-Processor & Brain Bridge (Phase 17 Capstone Finale)
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

from src.brain_bridge_motoru import (
    MotorDecodingPathway,
    SensoryFeedbackPathway,
    NeuromorphicBioCoprocessor,
)
from src.bridge_profilleyici import BridgeProfilleyici


def test_motor_decoding_pathway_output():
    """
    Motor Yolu Nöronal Çözümleme Testi.
    """
    path = MotorDecodingPathway(n_channels=64)
    spikes = np.random.rand(64)
    angle = path.decode_joint_angle(spikes)
    
    assert 0.0 <= angle <= 180.0


def test_sensory_feedback_pathway_pattern():
    """
    Duyusal Yolu Optogenetik Uyarım Deseni Testi.
    """
    path = SensoryFeedbackPathway(grid_size=(8, 8))
    pattern = path.generate_optogenetic_stimulus(pressure_val=5.0)
    
    assert pattern.shape == (8, 8)
    assert np.max(pattern) > 0.0


def test_neuromorphic_bio_coprocessor_closed_loop():
    """
    Çift Yönlü Kapalı Döngü Çalıştırma Testi.
    """
    coprocessor = NeuromorphicBioCoprocessor(n_channels=64)
    spikes = np.random.rand(64)
    res = coprocessor.run_closed_loop_cycle(spikes, tactile_pressure=7.0)
    
    assert "decoded_angle_deg" in res
    assert res["total_loop_ms"] < 1.0
    assert res["crypto_status"] == "AEAD_AUTHENTICATED"


def test_bridge_profiler_capstone_score():
    """
    FAZ 17 Capstone Final Profilleyici Metrik Doğrulaması.
    """
    metrics = BridgeProfilleyici.profille(motor_accuracy_pct=98.5, sensory_fidelity_pct=99.0, loop_latency_ms=0.12)
    
    assert metrics["phase17_capstone_score"] == 100.0
    assert metrics["motor_accuracy_score"] == 98.5
