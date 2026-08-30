"""
Day 330: Dendritic Computation & Non-linear Pyramidal Branch Dynamics
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

from src.dendritic_pyramidal_motoru import (
    DendriticBranch,
    MultiCompartmentPyramidalNeuron,
    DendriticXORClassifier,
)
from src.dendritic_profilleyici import DendriticProfilleyici


def test_dendritic_branch_nmda_spike():
    """
    Dendritik Dal NMDA Plateau Potansiyeli ve Eşikleme Testi.
    """
    branch = DendriticBranch(branch_id=1, num_synapses=2, threshold=1.0, plateau_gain=2.5)
    
    # Eşik altı girdi
    v_sub, is_sub = branch.compute_branch_potential(np.array([0.3, 0.3], dtype=np.float32))
    assert not is_sub
    assert v_sub < 1.0

    # Eşik üstü girdi (NMDA Spike)
    v_super, is_super = branch.compute_branch_potential(np.array([0.8, 0.8], dtype=np.float32))
    assert is_super
    assert v_super > 1.2


def test_multi_compartment_pyramidal_neuron_cable():
    """
    Çok Bölmeli Piramidal Nöron Kablo Entegrasyon Testi.
    """
    neuron = MultiCompartmentPyramidalNeuron(v_rest=-70.0, v_th=-50.0, g_coupling=0.5)
    in_b1 = np.array([1.0, 1.0], dtype=np.float32)
    in_b2 = np.array([1.0, 1.0], dtype=np.float32)
    
    v_soma, is_spike, states = neuron.step_simulation(in_b1, in_b2)
    assert "v_soma" in states
    assert "v_basal1" in states
    assert "v_basal2" in states


def test_dendritic_xor_classifier_all_cases():
    """
    Tek Piramidal Nöron ile %100 Doğru XOR Çözüm Testi.
    """
    classifier = DendriticXORClassifier()
    
    assert classifier.predict_xor(0.0, 0.0) == 0
    assert classifier.predict_xor(0.0, 1.0) == 1
    assert classifier.predict_xor(1.0, 0.0) == 1
    assert classifier.predict_xor(1.0, 1.0) == 0


def test_dendritic_profiler_metrics():
    """
    Dendritik Profilleyici Metrik Doğrulaması.
    """
    metrics = DendriticProfilleyici.profille(
        nmda_spikes_count=45,
        xor_accuracy=100.0,
        capacity_gain_x=4.0
    )
    
    assert metrics["xor_accuracy"] == 100.0
    assert metrics["capacity_gain_x"] == 4.0
    assert metrics["dendritic_capacity_score"] > 90.0
