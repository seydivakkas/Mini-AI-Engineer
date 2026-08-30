"""
Day 332: Optogenetic Stimulus Pattern Synthesis & Generative Inversion
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Birim Test Paketi (PyTest Suite)
"""

import sys
import os
import pytest
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.optogenetic_sentez_motoru import (
    ChR2OpsinModel,
    OptogeneticNeuralPopulation,
    OptogeneticGenerativeInverter,
)
from src.optogenetic_profilleyici import OptogeneticProfilleyici


def test_chr2_opsin_model_photocurrent():
    """
    ChR2 Opsin Fotoakım Hesabı Doğrulaması.
    """
    opsin = ChR2OpsinModel(g_max=0.4, i_sat=2.0)
    
    # Sıfır ışıkta sıfır fotoakım
    i0 = opsin.compute_photocurrent(light_irradiance=0.0)
    assert i0 == 0.0

    # Işık şiddeti ile negatif depolarize edici fotoakım (Inward current)
    i1 = opsin.compute_photocurrent(light_irradiance=3.0, v_membrane=-70.0)
    assert i1 < 0.0


def test_optogenetic_neural_population_simulation():
    """
    Optogenetik Nöromorfik Doku Simülasyon Testi.
    """
    population = OptogeneticNeuralPopulation(num_neurons=10)
    light_pattern = np.full(10, 4.0, dtype=np.float32)
    
    spikes, v_mem = population.simulate_step(light_pattern)
    assert len(spikes) == 10
    assert len(v_mem) == 10


def test_optogenetic_generative_inverter_synthesis():
    """
    Üretken İnversiyon Optimizer Kayıp Azalma Testi.
    """
    inverter = OptogeneticGenerativeInverter(num_neurons=10, time_steps=10)
    target = torch.eye(10, dtype=torch.float32)
    
    optimal_light, loss_history = inverter.sentezle_isik_deseni(target, num_epochs=15, lr=0.05)
    
    assert optimal_light.shape == (10, 10)
    assert len(loss_history) == 15
    # Son kayıp ilk kayıptan küçük olmalı
    assert loss_history[-1] < loss_history[0]


def test_optogenetic_profiler_metrics():
    """
    Optogenetik Profilleyici Metrik Doğrulaması.
    """
    metrics = OptogeneticProfilleyici.profille(
        max_light_irradiance=2.5,
        final_loss=0.025,
        reconstruction_fidelity=95.0
    )
    
    assert metrics["phototoxicity_safety_score"] > 70.0
    assert metrics["optogenetic_readiness_score"] > 80.0
