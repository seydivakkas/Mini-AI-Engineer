"""
Day 321: Spiking Neural Networks (SNN) & Leaky Integrate-and-Fire (LIF) Neuron Mathematics
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Birim Test Paketi (PyTest Suite)
"""

import sys
import os
import pytest
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.lif_snn_motoru import (
    FastSigmoidSurrogate,
    LIFNeuronCell,
    LIFSpikingLayer,
    SNNClassifier,
    PoissonEncoder,
)
from src.snn_profilleyici import SNNProfilleyici


def test_surrogate_gradient_backward():
    """
    FastSigmoidSurrogate ileri ve geri geçiş doğrulaması.
    """
    v_mem = torch.tensor([0.5, 1.2, 0.9], requires_grad=True)
    v_th = 1.0
    
    spikes = FastSigmoidSurrogate.apply(v_mem, v_th)
    assert torch.equal(spikes, torch.tensor([0.0, 1.0, 0.0]))
    
    loss = torch.sum(spikes)
    loss.backward()
    
    assert v_mem.grad is not None
    assert v_mem.grad.shape == v_mem.shape
    assert torch.all(v_mem.grad > 0)  # Surrogate gradyan sıfır olmamalı


def test_lif_neuron_cell_decay_and_fire():
    """
    LIFNeuronCell zar sızıntısı ve eşik aşımı spike testi.
    """
    cell = LIFNeuronCell(num_neurons=2, beta=0.5, v_threshold=1.0, v_reset=0.0)
    state = cell.init_state(batch_size=1, device=torch.device("cpu"))
    
    # 1. Adım: Düşük akım -> Spike üretmemeli, potansiyel artmalı
    current1 = torch.tensor([[0.8, 0.2]])
    spikes1, state1 = cell(current1, state)
    assert torch.equal(spikes1, torch.tensor([[0.0, 0.0]]))
    
    # 2. Adım: Yüksek akım -> 1. nöron eşiği aşmalı ve spike vermeli
    current2 = torch.tensor([[1.5, 0.1]])
    spikes2, state2 = cell(current2, state1)
    assert spikes2[0, 0].item() == 1.0
    # Spike veren 1. nöron sıfırlanmalı (V_reset=0.0)
    assert state2[0][0, 0].item() == 0.0


def test_lif_refractory_period():
    """
    Ateşleme sonrası refrakter süre boyunca nöronun kilitli kalması testi.
    """
    ref_period = 3
    cell = LIFNeuronCell(num_neurons=1, beta=0.8, v_threshold=1.0, v_reset=0.0, refractory_period=ref_period)
    state = cell.init_state(batch_size=1, device=torch.device("cpu"))
    
    # Spike üret
    current = torch.tensor([[2.0]])
    spikes, state = cell(current, state)
    assert spikes[0, 0].item() == 1.0
    
    # Sonraki ref_period adımda akım yüksek olsa dahi kilitli kalmalı
    for _ in range(ref_period):
        spikes_ref, state = cell(current, state)
        assert spikes_ref[0, 0].item() == 0.0
        assert state[0][0, 0].item() == 0.0  # V_reset'te sabit


def test_poisson_encoder():
    """
    PoissonEncoder çıktı boyut ve değer aralığı testi.
    """
    time_steps = 30
    encoder = PoissonEncoder(time_steps=time_steps)
    x = torch.tensor([[0.2, 0.8], [0.5, 0.0]])
    
    spikes = encoder(x)
    assert spikes.shape == (2, time_steps, 2)
    assert torch.all((spikes == 0.0) | (spikes == 1.0))


def test_lif_spiking_layer_shape():
    """
    LIFSpikingLayer katman boyutları doğrulaması.
    """
    layer = LIFSpikingLayer(in_features=10, out_features=5, beta=0.8)
    x_seq = torch.rand(4, 20, 10)  # (Batch=4, T=20, In=10)
    
    spikes_seq, mem_seq = layer(x_seq)
    assert spikes_seq.shape == (4, 20, 5)
    assert mem_seq.shape == (4, 20, 5)


def test_snn_classifier_end_to_end():
    """
    SNNClassifier uçtan uca ileri geçiş ve backprop gradyan testi.
    """
    model = SNNClassifier(in_features=16, hidden_features=32, num_classes=3, time_steps=15)
    x = torch.rand(2, 16)
    y = torch.tensor([0, 2])
    
    logits, info_dict = model(x)
    assert logits.shape == (2, 3)
    assert "spikes1" in info_dict
    assert "sparsity_layer1" in info_dict
    
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    
    # Katman parametrelerinin gradyanları hesaplanmış olmalı
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_snn_profillexer():
    """
    SNNProfilleyici metrik hesaplama testi.
    """
    model = SNNClassifier(in_features=8, hidden_features=16, num_classes=2, time_steps=10)
    x = torch.rand(4, 8)
    logits, info_dict = model(x)
    
    metrics = SNNProfilleyici.profille(model, info_dict, time_steps=10)
    assert "total_sops" in metrics
    assert "snn_energy_pj" in metrics
    assert "energy_gain_x" in metrics
    assert metrics["snn_energy_pj"] >= 0.0
