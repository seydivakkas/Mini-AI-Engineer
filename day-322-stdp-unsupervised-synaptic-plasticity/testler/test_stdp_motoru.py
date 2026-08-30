"""
Day 322: Spike-Timing-Dependent Plasticity (STDP) & Unsupervised Learning
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Birim Test Paketi (PyTest Suite)
"""

import sys
import os
import pytest
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.stdp_motoru import (
    STDPLearningRule,
    WTALateralInhibition,
    STDPUnsupervisedNetwork,
)
from src.stdp_profilleyici import STDPProfilleyici


def test_stdp_ltp_potentiation():
    """
    LTP (Long-Term Potentiation): Presinaptik spike -> Postsinaptik spike
    Ağırlığın artması gerektiğini doğrular.
    """
    stdp = STDPLearningRule(a_plus=0.1, a_minus=0.0, learning_rate=0.1, w_min=0.0, w_max=1.0)
    w = torch.tensor([[0.5]])
    
    # 1. Adım: Presinaptik spike (Pre=1, Post=0) -> Iz birikir
    s_pre1 = torch.tensor([[1.0]])
    s_post1 = torch.tensor([[0.0]])
    t_pre1, t_post1 = stdp.init_traces(1, 1, 1, torch.device("cpu"))
    
    w1, t_pre1, t_post1, _ = stdp.update_weights(w, s_pre1, s_post1, t_pre1, t_post1)
    
    # 2. Adım: Postsinaptik spike (Pre=0, Post=1) -> LTP tetiklenir
    s_pre2 = torch.tensor([[0.0]])
    s_post2 = torch.tensor([[1.0]])
    
    w2, _, _, delta_w = stdp.update_weights(w1, s_pre2, s_post2, t_pre1, t_post1)
    
    assert w2[0, 0].item() > 0.5  # Ağırlık artmalı (LTP)


def test_stdp_ltd_depression():
    """
    LTD (Long-Term Depression): Postsinaptik spike -> Presinaptik spike
    Ağırlığın azalması gerektiğini doğrular.
    """
    stdp = STDPLearningRule(a_plus=0.0, a_minus=0.1, learning_rate=0.1, w_min=0.0, w_max=1.0)
    w = torch.tensor([[0.5]])
    
    # 1. Adım: Postsinaptik spike (Pre=0, Post=1)
    s_pre1 = torch.tensor([[0.0]])
    s_post1 = torch.tensor([[1.0]])
    t_pre1, t_post1 = stdp.init_traces(1, 1, 1, torch.device("cpu"))
    
    w1, t_pre1, t_post1, _ = stdp.update_weights(w, s_pre1, s_post1, t_pre1, t_post1)
    
    # 2. Adım: Presinaptik spike (Pre=1, Post=0) -> LTD tetiklenir
    s_pre2 = torch.tensor([[1.0]])
    s_post2 = torch.tensor([[0.0]])
    
    w2, _, _, _ = stdp.update_weights(w1, s_pre2, s_post2, t_pre1, t_post1)
    
    assert w2[0, 0].item() < 0.5  # Ağırlık azalmalı (LTD)


def test_stdp_weight_clamping():
    """
    Ağırlıkların w_min ve w_max sınırları dışına çıkmadığını doğrular.
    """
    stdp = STDPLearningRule(a_plus=5.0, a_minus=0.0, learning_rate=1.0, w_min=0.0, w_max=1.0)
    w = torch.tensor([[0.9]])
    
    s_pre = torch.tensor([[1.0]])
    s_post = torch.tensor([[1.0]])
    t_pre, t_post = stdp.init_traces(1, 1, 1, torch.device("cpu"))
    
    w_new, _, _, _ = stdp.update_weights(w, s_pre, s_post, t_pre, t_post)
    assert w_new[0, 0].item() == 1.0  # w_max sınırına takılmalı


def test_wta_lateral_inhibition():
    """
    Winner-Take-All yanal inhibisyonun yalnızca 1 kazanan spike bırakmasını doğrular.
    """
    wta = WTALateralInhibition()
    spikes = torch.tensor([[0.2, 0.9, 0.5]])  # 2. nöron en yüksek
    
    wta_output = wta(spikes)
    assert torch.equal(wta_output, torch.tensor([[0.0, 0.9, 0.0]]))


def test_stdp_unsupervised_network_forward():
    """
    STDPUnsupervisedNetwork uçtan uca ileri geçiş doğrulaması.
    """
    net = STDPUnsupervisedNetwork(in_features=8, out_features=2, time_steps=10)
    spikes_seq = torch.rand(4, 10, 8)
    
    out_dict = net(spikes_seq, train_stdp=True)
    assert "spikes_seq" in out_dict
    assert "final_weights" in out_dict
    assert out_dict["spikes_seq"].shape == (4, 10, 2)


def test_stdp_profiler():
    """
    STDPProfilleyici metrik çıktılarının doğrulaması.
    """
    w_init = torch.rand(2, 4).numpy()
    w_final = torch.rand(2, 4).numpy()
    spikes_seq = torch.rand(2, 10, 2)
    
    metrics = STDPProfilleyici.profille(w_init, w_final, spikes_seq)
    assert "mean_weight_drift" in metrics
    assert "bimodality_score" in metrics
    assert metrics["mean_weight_drift"] >= 0.0
