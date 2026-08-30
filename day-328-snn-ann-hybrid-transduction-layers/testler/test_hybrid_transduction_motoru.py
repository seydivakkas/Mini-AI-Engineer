"""
Day 328: SNN-ANN Hybrid Transduction Layers
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

from src.hybrid_transduction_motoru import (
    ANNToSNNTransducer,
    SNNToANNTransducer,
    SNNLIFLayer,
    HybridSNNANNNetwork,
)


def test_ann_to_snn_transducer_shape():
    """
    ANN-to-SNN Transducer tensor boyutu ve Spike (0 veya 1) doğrulaması.
    """
    batch_size = 4
    n_ann = 16
    n_snn = 32
    time_steps = 10
    
    transducer = ANNToSNNTransducer(in_features=n_ann, out_features=n_snn, time_steps=time_steps)
    x_ann = torch.randn(batch_size, n_ann)
    spike_stream = transducer(x_ann)
    
    assert spike_stream.shape == (batch_size, time_steps, n_snn)
    # Sadece 0.0 veya 1.0 olmalı
    unique_vals = torch.unique(spike_stream)
    assert all(val in [0.0, 1.0] for val in unique_vals.tolist())


def test_snn_to_ann_transducer_shape():
    """
    SNN-to-ANN Transducer süzgeçli vektör boyutu doğrulaması.
    """
    batch_size = 4
    n_snn = 32
    n_ann = 16
    time_steps = 10
    
    transducer = SNNToANNTransducer(in_features=n_snn, out_features=n_ann, tau_decay=2.0)
    spike_stream = (torch.rand(batch_size, time_steps, n_snn) > 0.7).float()
    h_ann = transducer(spike_stream)
    
    assert h_ann.shape == (batch_size, n_ann)
    assert torch.all(h_ann >= 0.0)  # ReLU output


def test_snn_lif_layer_integration():
    """
    SNN LIF Katmanı zar potansiyeli entegrasyonu ve spike tetikleme testi.
    """
    batch_size = 2
    time_steps = 5
    num_neurons = 8
    
    lif_layer = SNNLIFLayer(num_neurons=num_neurons, beta=0.8, v_th=1.0)
    input_spikes = torch.ones(batch_size, time_steps, num_neurons)  # Sürekli spike girdisi
    
    out_spikes, v_mem_history = lif_layer(input_spikes)
    
    assert out_spikes.shape == (batch_size, time_steps, num_neurons)
    assert v_mem_history.shape == (batch_size, time_steps, num_neurons)
    # En az bir adımda spike atmalı
    assert out_spikes.sum() > 0.0


def test_hybrid_network_forward_and_backward():
    """
    Hibrit SNN-ANN ağının tam ileri (forward) ve geri (backward) türev akışı testi.
    """
    batch_size = 4
    in_features = 32
    num_classes = 3
    
    model = HybridSNNANNNetwork(in_features=in_features, ann_hidden=16, snn_neurons=16, num_classes=num_classes, time_steps=6)
    x_input = torch.randn(batch_size, in_features)
    
    logits, spike_stream, v_mem = model(x_input)
    loss = logits.sum()
    loss.backward()
    
    assert logits.shape == (batch_size, num_classes)
    assert spike_stream.shape == (batch_size, 6, 16)
    assert model.classifier.weight.grad is not None
