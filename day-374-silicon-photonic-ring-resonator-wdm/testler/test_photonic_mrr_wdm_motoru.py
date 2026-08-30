"""
Day 374: Silicon Photonic Micro-Ring Resonator and WDM Weight Bank
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

from src.photonic_mrr_wdm_motoru import (
    MicroRingResonator,
    WDMWeightBankCrossbar,
    PhotonicWDMBenchmark,
)
from src.mrr_wdm_profilleyici import MRRWDMProfilleyici


def test_micro_ring_resonator_transmission():
    """
    Mikro-Halka Rezonatör Geçirgenlik Modeli Testi.
    """
    mrr = MicroRingResonator(radius_um=8.0)
    t0 = mrr.get_transmission(1545.0, delta_temp_k=0.0)
    t_hot = mrr.get_transmission(1545.0, delta_temp_k=5.0)
    
    assert 0.0 <= t0 <= 1.0
    assert 0.0 <= t_hot <= 1.0


def test_wdm_weight_bank_dot_product():
    """
    WDM Ağırlık Bankası Nokta Çarpım Testi.
    """
    bank = WDMWeightBankCrossbar(num_channels=16)
    w = np.full(16, 0.5)
    x = np.full(16, 1.0)
    bank.program_weights(w)
    dp_res, trans = bank.compute_dot_product(x)
    
    assert len(trans) == 16
    assert dp_res > 0.0


def test_photonic_wdm_benchmark():
    """
    Fotonik WDM Kıyaslama Doğruluk Testi.
    """
    bench = PhotonicWDMBenchmark()
    res = bench.run_benchmark()
    
    assert res["cosine_fidelity"] >= 0.98
    assert res["crosstalk_db"] <= -25.0
    assert res["throughput_tbps"] >= 1.0


def test_mrr_wdm_profiler_metrics():
    """
    WDM Profilleyici Metrik Testi.
    """
    mock_res = {
        "ideal_dot_prod": 5.2,
        "photonic_dot_prod": 5.18,
        "cosine_fidelity": 0.998,
        "crosstalk_db": -29.2,
        "throughput_tbps": 1.6
    }
    metrics = MRRWDMProfilleyici.profille(mock_res)
    assert metrics["fidelity_score"] >= 99.0
    assert metrics["wdm_readiness_score"] >= 98.0
