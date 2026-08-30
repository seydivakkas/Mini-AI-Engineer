"""
Day 344: Radiation-Hardened Fault-Tolerant Edge AI Inference with Triple Modular Redundancy (TMR)
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

from src.rad_hard_tmr_motoru import (
    RadiationSEUInjector,
    TMRInferenceCore,
    AutonomousMemoryScrubber,
    FaultTolerantAIEngine,
)
from src.tmr_profilleyici import TMRProfilleyici


def test_radiation_seu_injection():
    """
    Kozmik Radyasyon SEU Bit-Flip Enjeksiyon Testi.
    """
    injector = RadiationSEUInjector()
    w = np.ones((4, 4), dtype=np.float32)
    corrupted = injector.inject_seu_to_weights(w, num_flips=2)
    
    assert not np.array_equal(w, corrupted)


def test_tmr_majority_voting_consensus():
    """
    TMR 2/3 Çoğunluk Oylaması ve Hatalı Çekirdek İzolasyon Testi.
    """
    golden_w = np.eye(4)
    tmr = TMRInferenceCore(golden_w)
    
    # Core B'yi farklı sınıf üretecek şekilde boz
    tmr.core_b_weights = np.roll(np.eye(4), 1, axis=1)
    
    x = np.array([[1.0, 0.0, 0.0, 0.0]])
    res = tmr.tmr_inference(x)
    
    assert res["has_fault"] is True
    assert "Core_B" in res["faulty_cores"]
    assert res["majority_pred"] == 0  # Core A ve Core C çoğunluğu


def test_autonomous_scrubber_repair():
    """
    Otonom Bellek Temizleme ve Onarım Testi.
    """
    golden_w = np.eye(4)
    tmr = TMRInferenceCore(golden_w)
    scrubber = AutonomousMemoryScrubber(tmr)
    
    tmr.core_b_weights = np.zeros((4, 4))
    scrubber.scrub_and_repair(["Core_B"])
    
    assert np.array_equal(tmr.core_b_weights, golden_w)
    assert scrubber.repair_count == 1


def test_tmr_profiler_metrics():
    """
    TMR Profilleyici Metrik Doğrulama Testi.
    """
    metrics = TMRProfilleyici.profille(
        single_core_accuracy=80.0,
        tmr_accuracy=100.0,
        total_seu_events=10,
        repaired_events=10
    )
    
    assert metrics["seu_recovery_rate"] == 100.0
    assert metrics["tmr_accuracy"] == 100.0
    assert metrics["space_rad_hard_score"] > 99.0
