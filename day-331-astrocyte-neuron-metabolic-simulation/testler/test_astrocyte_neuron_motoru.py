"""
Day 331: Astrocyte-Neuron Metabolic Interaction & Slow Neuromodulation
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

from src.astrocyte_neuron_motoru import (
    AstrocyteCalciumModel,
    TripartiteSynapse,
    AstrocyteMetabolicNetwork,
)
from src.astrocyte_profilleyici import AstrocyteProfilleyici


def test_astrocyte_calcium_model_update():
    """
    Astrosit İçi Kalsiyum Dinamikleri ve Eşikleme Testi.
    """
    astro = AstrocyteCalciumModel(ca_rest=0.05, theta_ca=0.35)
    
    # Düşük girdi
    ca1, glio1 = astro.update_calcium(glutamate_input=0.0)
    assert ca1 <= 0.05
    assert glio1 == 0.0

    # Yüksek girdi (Kalsiyum dalgası)
    for _ in range(10):
        ca2, glio2 = astro.update_calcium(glutamate_input=2.0)

    assert ca2 > 0.35
    assert glio2 > 0.5


def test_tripartite_synapse_modulation():
    """
    Üçlü Sinaps Yavaş Nöromodülasyon (P_release Artışı) Testi.
    """
    synapse = TripartiteSynapse(p_base=0.4)
    p_initial = synapse.current_p_release
    
    # Sürekli presinaptik uyarım ver
    for _ in range(15):
        _, _, p_mod = synapse.step_synapse(presynaptic_spike=True)

    assert p_mod > p_initial


def test_astrocyte_metabolic_network_simulation():
    """
    ANLS Astrosit Metabolik Ağ Adım Simülasyonu Testi.
    """
    network = AstrocyteMetabolicNetwork(num_neurons=8)
    spikes = np.ones(8, dtype=np.bool_)
    
    res = network.simulate_step(spikes)
    assert "transmitted_count" in res
    assert "mean_ca" in res
    assert "mean_p_release" in res
    assert "mean_atp" in res
    assert res["mean_atp"] > 0.0


def test_astrocyte_profiler_metrics():
    """
    Astrosit Profilleyici Metrik Doğrulaması.
    """
    metrics = AstrocyteProfilleyici.profille(
        ca_spikes_count=12,
        mean_p_release=0.65,
        mean_atp_level=95.0
    )
    
    assert metrics["ca_spikes_count"] == 12
    assert metrics["tripartite_readiness_score"] > 85.0
