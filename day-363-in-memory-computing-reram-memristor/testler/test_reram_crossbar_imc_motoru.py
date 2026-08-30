"""
Day 363: In-Memory Computing (IMC) with ReRAM & Memristor Crossbar Arrays
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

from src.reram_crossbar_imc_motoru import (
    MemristorCell,
    DifferentialReRAMCrossbar,
    InStorageAnalogVMMProcessor,
    ReRAMInferenceBenchmark,
)
from src.reram_profilleyici import ReRAMProfilleyici


def test_memristor_cell_conductance_range():
    """
    Memristör İletkenlik Aralığı Testi.
    """
    cell = MemristorCell(g_min_us=10.0, g_max_us=200.0)
    cell.set_conductance(100e-6, noise_std=0.0)
    assert 10e-6 <= cell.g <= 200e-6


def test_differential_reram_crossbar_program():
    """
    Diferansiyel ReRAM Ağırlık Programlama Testi.
    """
    crossbar = DifferentialReRAMCrossbar(rows=8, cols=8)
    w_mat = np.random.uniform(-1.0, 1.0, (8, 8))
    crossbar.program_weights(w_mat)
    
    assert crossbar.g_pos.shape == (8, 8)
    assert crossbar.g_neg.shape == (8, 8)
    assert (crossbar.g_pos >= crossbar.g_min).all()
    assert (crossbar.g_neg >= crossbar.g_min).all()


def test_in_storage_analog_vmm_compute():
    """
    Bellek İçi Analog VMM Hesaplama Testi.
    """
    proc = InStorageAnalogVMMProcessor(rows=8, cols=8)
    w_mat = np.eye(8)
    proc.crossbar.program_weights(w_mat)
    
    v_in = np.ones(8)
    y_out = proc.compute(v_in)
    assert y_out.shape == (8,)
    assert not np.isnan(y_out).any()


def test_reram_profiler_metrics():
    """
    ReRAM Profilleyici Metrik Testi.
    """
    mock_res = {
        "fidelity_score": 98.5,
        "energy_efficiency_gain": 18.7,
        "analog_compute_latency_ns": 3.2
    }
    metrics = ReRAMProfilleyici.profille(mock_res)
    assert metrics["fidelity_score"] == 98.5
    assert metrics["reram_readiness"] > 98.0
