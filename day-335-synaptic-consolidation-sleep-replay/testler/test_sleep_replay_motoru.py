"""
Day 335: Synaptic Consolidation & Sleep Replay (Zero Catastrophic Forgetting)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Birim Test Paketi (PyTest Suite)
"""

import sys
import os
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.sleep_replay_motoru import (
    SynapticTaggingConsolidator,
    HippocampalSleepReplayer,
    ContinualSpikingNetwork,
)
from src.sleep_profilleyici import SleepProfilleyici


def test_synaptic_tagging_consolidator_fisher():
    """
    Fisher Information ve Konsolidasyon Kayıp Doğrulaması.
    """
    model = ContinualSpikingNetwork(input_dim=10, hidden_dim=16, num_classes=2)
    x = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    loader = DataLoader(TensorDataset(x, y), batch_size=10)
    criterion = nn.CrossEntropyLoss()

    consolidator = SynapticTaggingConsolidator(model, lambda_cons=100.0)
    consolidator.compute_fisher_information(loader, criterion)
    
    assert len(consolidator.fisher_dict) > 0
    loss_cons = consolidator.consolidation_loss()
    assert loss_cons.item() == 0.0  # İlk durumda 0 olmalı


def test_hippocampal_sleep_replayer_buffer():
    """
    Uyku Fazı Bellek Tekrarı Tamponu Testi.
    """
    replayer = HippocampalSleepReplayer(capacity=50)
    x = torch.randn(10, 8)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    for i in range(10):
        replayer.store_wake_memory(x[i], y[i])

    rx, ry = replayer.sample_sleep_replay(batch_size=4)
    assert len(rx) == 4
    assert len(ry) == 4


def test_continual_spiking_network_forward():
    """
    Sürekli Öğrenme Ağ İleri Besleme Testi.
    """
    model = ContinualSpikingNetwork(input_dim=16, hidden_dim=32, num_classes=4)
    inputs = torch.randn(8, 16)
    outputs = model(inputs)
    
    assert outputs.shape == (8, 4)


def test_sleep_profiler_metrics():
    """
    Uyku Konsolidasyon Profilleyici Metrik Doğrulaması.
    """
    metrics = SleepProfilleyici.profille(
        task1_retention=97.5,
        task2_accuracy=94.0,
        forgetting_std=60.0,
        forgetting_sleep=0.0
    )
    
    assert metrics["task1_retention_score"] == 97.5
    assert metrics["zero_forgetting_readiness_score"] > 90.0
