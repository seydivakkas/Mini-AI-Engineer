"""
Day 375: Photonic Spiking Neural Network with Picosecond Spike Processing
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

from src.photonic_snn_motoru import (
    PhotonicIntegrateAndFireNeuron,
    PhotonicWaveguideSynapse,
    PhotonicSpikingNetwork,
    PhotonicSNNBenchmark,
)
from src.photonic_snn_profilleyici import PhotonicSNNProfilleyici


def test_photonic_if_neuron_spike():
    """
    Fotonik IF Nöron Entegrasyon ve Ateşleme Testi.
    """
    neuron = PhotonicIntegrateAndFireNeuron(v_thresh=1.0)
    spk1 = neuron.step(0.5, dt_ps=10.0, current_time_ps=10.0)
    assert not spk1
    assert neuron.v_mem > 0.4
    
    spk2 = neuron.step(0.7, dt_ps=10.0, current_time_ps=20.0)
    assert spk2
    assert neuron.v_mem == 0.0 # Sıfırlandı


def test_photonic_waveguide_synapse_stdp():
    """
    Optik STDP Sinaptik Plastisite Testi.
    """
    syn = PhotonicWaveguideSynapse(init_weight=0.5)
    init_w = syn.w
    # Pre önce, Post sonra -> LTP (Ağırlık artar)
    syn.update_optical_stdp(t_pre_ps=100.0, t_post_ps=150.0)
    assert syn.w > init_w
    
    # Post önce, Pre sonra -> LTD (Ağırlık azalır)
    syn.update_optical_stdp(t_pre_ps=200.0, t_post_ps=150.0)
    assert syn.w < 0.95


def test_photonic_snn_benchmark():
    """
    Fotonik SNN Kıyaslama Motoru Testi.
    """
    bench = PhotonicSNNBenchmark()
    res = bench.run_benchmark()
    
    assert res["spike_rate_ghz"] >= 20.0
    assert res["energy_pj_per_spike"] <= 0.20
    assert res["pattern_accuracy"] > 95.0


def test_photonic_snn_profiler_metrics():
    """
    Fotonik SNN Profilleyici Metrik Testi.
    """
    mock_res = {
        "spike_rate_ghz": 20.0,
        "energy_pj_per_spike": 0.15,
        "pattern_accuracy": 98.8
    }
    metrics = PhotonicSNNProfilleyici.profille(mock_res)
    assert metrics["rate_score"] >= 99.0
    assert metrics["snn_readiness_score"] >= 98.0
