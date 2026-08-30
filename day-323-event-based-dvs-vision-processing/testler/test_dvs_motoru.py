"""
Day 323: Dynamic Vision Sensors (DVS) & Event-Based Processing
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

from src.dvs_motoru import (
    DVSEventStreamGenerator,
    SurfaceOfActiveEvents,
    VoxelGridEncoder,
    SpikingEventConvNet,
)
from src.dvs_profilleyici import DVSProfilleyici


def test_dvs_event_stream_generator():
    """
    DVSEventStreamGenerator olay üretimi doğrulaması.
    """
    gen = DVSEventStreamGenerator(height=32, width=32)
    events = gen.uret_hareketli_cizgi(yon="sag", duration_us=10000, num_events=100)
    
    assert events.shape == (100, 4)
    assert np.all(events[:, 0] >= 0) and np.all(events[:, 0] < 32)
    assert np.all(events[:, 1] >= 0) and np.all(events[:, 1] < 32)
    assert np.all(np.isin(events[:, 3], [-1.0, 1.0]))


def test_surface_of_active_events_decay():
    """
    SAE yüzeyi üstel sönümlenme testi.
    """
    sae = SurfaceOfActiveEvents(height=16, width=16, tau_us=1000.0)
    events = np.array([[5, 5, 1000.0, 1.0]], dtype=np.float32)
    
    # Anlık hesaplama (t = 1000) -> sönümlenme exp(0) = 1.0
    surf1 = sae.guncelle_ve_hesapla(events, t_current=1000.0)
    assert np.isclose(surf1[1, 5, 5], 1.0)
    
    # Sonraki zaman (t = 2000 -> tau kadar sonra) -> exp(-1) ~= 0.3678
    surf2 = sae.guncelle_ve_hesapla(np.empty((0, 4)), t_current=2000.0)
    assert np.isclose(surf2[1, 5, 5], np.exp(-1.0), atol=1e-3)


def test_voxel_grid_encoder_shape():
    """
    VoxelGridEncoder çıktı boyut ve kanal doğrulaması.
    """
    num_bins = 4
    encoder = VoxelGridEncoder(height=20, width=20, num_bins=num_bins)
    events = np.array([
        [10, 10, 500.0, 1.0],
        [12, 14, 800.0, -1.0]
    ], dtype=np.float32)
    
    grid = encoder.kodla(events, duration_us=1000.0)
    assert grid.shape == (2 * num_bins, 20, 20)
    assert torch.sum(grid) == 2.0


def test_spiking_event_convnet_forward():
    """
    SpikingEventConvNet ileri ve geri geçiş testi.
    """
    net = SpikingEventConvNet(in_channels=6, num_classes=3)
    x = torch.rand(2, 6, 16, 16)
    
    logits, info = net(x)
    assert logits.shape == (2, 3)
    assert "spikes_l1" in info
    
    loss = torch.sum(logits)
    loss.backward()
    assert net.conv1.weight.grad is not None


def test_dvs_profiler():
    """
    DVSProfilleyici sıkıştırma oranı metrik testi.
    """
    events = np.random.rand(500, 4)
    metrics = DVSProfilleyici.profille(events, height=32, width=32, duration_us=50000.0)
    
    assert "compression_ratio_x" in metrics
    assert metrics["compression_ratio_x"] > 1.0
