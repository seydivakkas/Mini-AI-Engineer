"""
Day 346: Electronic Warfare (EW) Cognitive RF Spectrum Sensing & Jamming Mitigation
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

from src.cognitive_ew_motoru import (
    RFEmitterSimulator,
    CognitiveSpectrumClassifier,
    AdaptiveAntiJammingAgent,
)
from src.ew_profilleyici import EWProfilleyici


def test_rf_emitter_simulator():
    """
    RF Sinyal Üreteci Sentezleme Testi.
    """
    sim = RFEmitterSimulator(num_samples=256)
    for sig_type in RFEmitterSimulator.SIGNAL_TYPES:
        i_sig, q_sig = sim.generate_signal(sig_type)
        assert len(i_sig) == 256
        assert len(q_sig) == 256
        assert np.mean(i_sig**2 + q_sig**2) > 0.0


def test_cognitive_spectrum_classifier_features():
    """
    Bilişsel Spektrum Özellik Çıkarımı Testi.
    """
    sim = RFEmitterSimulator(num_samples=256)
    classifier = CognitiveSpectrumClassifier()
    
    i_sig, q_sig = sim.generate_signal("QPSK_COMM")
    feats = classifier.extract_rf_features(i_sig, q_sig)
    
    assert len(feats) == 6
    res = classifier.classify_emitter(i_sig, q_sig)
    assert "predicted_emitter" in res
    assert res["confidence"] > 0.8


def test_adaptive_anti_jamming_agent_learning():
    """
    Bilişsel Anti-Jamming Ajanı Öğrenme Testi.
    """
    agent = AdaptiveAntiJammingAgent(num_channels=4, epsilon=0.0) # Greedy
    
    # Kanal 0 karıştırıldı (Ceza)
    agent.update_channel_reward(channel=0, sinr_db=-5.0, is_jammed=True)
    # Kanal 2 temiz (Ödül)
    agent.update_channel_reward(channel=2, sinr_db=20.0, is_jammed=False)
    
    best_ch = agent.select_transmission_channel()
    assert best_ch == 2 # Temiz kanal seçilmeli


def test_ew_profiler_metrics():
    """
    Elektronik Harp Profilleyici Testi.
    """
    metrics = EWProfilleyici.profille(
        classification_accuracy=96.0,
        mean_sinr_db=18.5,
        jamming_collision_rate=0.05
    )
    
    assert metrics["threat_classification_score"] == 96.0
    assert metrics["anti_jamming_score"] == 90.0
    assert metrics["ew_dominance_score"] > 90.0
