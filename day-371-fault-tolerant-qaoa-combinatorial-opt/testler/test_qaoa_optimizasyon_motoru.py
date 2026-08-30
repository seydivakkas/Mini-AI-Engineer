"""
Day 371: Fault-Tolerant QAOA Quantum Circuit for Logistics Combinatorial Optimization
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

from src.qaoa_optimizasyon_motoru import (
    IsingCostHamiltonian,
    QAOACircuitSimulator,
    VariationalQuantumOptimizer,
    LogisticsQAOABenchmark,
)
from src.qaoa_profilleyici import QAOAProfilleyici


def test_ising_cost_hamiltonian_evaluation():
    """
    Ising Maliyet Hamiltonyeni Değerlendirme Testi.
    """
    h = IsingCostHamiltonian(num_qubits=5)
    cost_0 = h.evaluate_cost_for_bitstring(0) # |00000>
    assert cost_0 == 0.0 # Tüm spinler aynı yönde (Kesim yok)
    cost_max = max([h.evaluate_cost_for_bitstring(i) for i in range(32)])
    assert cost_max > 0.0


def test_qaoa_circuit_simulator_probabilities():
    """
    QAOA Devre Kuantum Olasılık Dağılımı Testi.
    """
    h = IsingCostHamiltonian(num_qubits=5)
    sim = QAOACircuitSimulator(h, p_layers=2)
    gamma = np.array([0.5, 0.5])
    beta = np.array([0.5, 0.5])
    probs = sim.simulate_state(gamma, beta)
    
    assert len(probs) == 32
    assert np.allclose(np.sum(probs), 1.0)


def test_logistics_qaoa_benchmark():
    """
    Lojistik QAOA Yaklaşım Oranı Testi.
    """
    bench = LogisticsQAOABenchmark()
    res = bench.run_benchmark()
    
    assert res["optimal_cost"] > 0.0
    assert res["approximation_ratio"] > 75.0 # En az %75 yaklaşım


def test_qaoa_profiler_metrics():
    """
    QAOA Profilleyici Metrik Testi.
    """
    mock_res = {
        "optimal_cost": 8.0,
        "qaoa_cost": 7.6,
        "approximation_ratio": 95.0,
        "optimal_prob": 0.25
    }
    metrics = QAOAProfilleyici.profille(mock_res)
    assert metrics["approximation_ratio"] == 95.0
    assert metrics["qaoa_readiness_score"] > 95.0
