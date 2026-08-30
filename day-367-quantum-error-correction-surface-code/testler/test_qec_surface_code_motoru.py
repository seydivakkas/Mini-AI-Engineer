"""
Day 367: Surface Code Quantum Error Correction (QEC) Neural Syndrome Decoder
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

from src.qec_surface_code_motoru import (
    PlanarSurfaceCodeLattice,
    QuantumNoiseChannel,
    NeuralSyndromeDecoder,
    QuantumErrorCorrectionBenchmark,
)
from src.qec_profilleyici import QECProfilleyici


def test_planar_surface_code_lattice_stabilizers():
    """
    Yüzey Kodu Stabilizatör ve Kafes Geometrisi Testi.
    """
    lattice = PlanarSurfaceCodeLattice(distance=3)
    assert lattice.num_data == 9
    assert lattice.H_X.shape == (4, 9)
    assert lattice.H_Z.shape == (4, 9)


def test_quantum_noise_channel_generation():
    """
    Depolarize Kuantum Gürültü Kanalı Testi.
    """
    noise = QuantumNoiseChannel(p_error=0.01)
    err_x, err_z = noise.generate_pauli_errors(num_qubits=9, batch_size=10)
    assert err_x.shape == (10, 9)
    assert err_z.shape == (10, 9)
    assert set(np.unique(err_x)).issubset({0, 1})


def test_neural_syndrome_decoder_inference():
    """
    Nöral Sendrom Dekoderi Çıkarım Testi.
    """
    decoder = NeuralSyndromeDecoder(num_syndromes=4, num_data=9)
    syndrome = np.array([1, 0, 1, 0])
    corr = decoder.decode(syndrome)
    assert corr.shape == (9,)
    assert decoder.inference_latency_ns < 100.0 # 100 ns altı


def test_qec_profiler_metrics():
    """
    QEC Profilleyici Metrik Testi.
    """
    mock_res = {
        "logical_fidelity": 0.995,
        "physical_fidelity": 0.950,
        "speedup": 160.0
    }
    metrics = QECProfilleyici.profille(mock_res)
    assert metrics["logical_fidelity"] == 99.5
    assert metrics["qec_readiness_score"] > 98.0
