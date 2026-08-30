"""
Day 329: Neuromorphic Auditory Cochlea Filters & Event-Based Acoustic Classification
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

from src.cochlea_audio_motoru import (
    GammatoneFilterBank,
    SiliconCochleaEventGenerator,
    SpikingAudioClassifier,
)
from src.cochlea_profilleyici import CochleaProfilleyici


def test_gammatone_filter_bank_center_freqs():
    """
    Gammatone Filtre Bankası ERB Frekans Dizilimi Doğrulaması.
    """
    bank = GammatoneFilterBank(num_channels=16, f_min=100.0, f_max=6000.0, fs=16000)
    
    assert len(bank.center_freqs) == 16
    assert bank.center_freqs[0] >= 95.0
    assert bank.center_freqs[-1] <= 6050.0
    # Frekanslar artan sırada olmalı
    assert np.all(np.diff(bank.center_freqs) > 0)


def test_silicon_cochlea_event_generator_shape():
    """
    Silikon Koklea Olay Tabanlı Spike Matrisi (Kokleogram) Boyut Doğrulaması.
    """
    bank = GammatoneFilterBank(num_channels=12, fs=16000)
    audio = np.sin(2 * np.pi * 400.0 * np.linspace(0, 0.5, 8000)).astype(np.float32)
    filtered = bank.filtrele(audio)
    
    event_gen = SiliconCochleaEventGenerator(threshold=0.05, time_bin_size=160)
    cochleogram, event_list = event_gen.uret_kokleogram_spikelari(filtered)
    
    expected_bins = 8000 // 160  # 50 bins
    assert cochleogram.shape == (12, expected_bins)
    assert len(event_list) > 0


def test_spiking_audio_classifier_forward():
    """
    SNN Akustik Komut Sınıflandırıcısı İleri Besleme (Forward Pass) Testi.
    """
    batch_size = 4
    num_channels = 16
    num_time_bins = 50
    num_classes = 4
    
    model = SpikingAudioClassifier(num_channels=num_channels, num_time_bins=num_time_bins, num_classes=num_classes)
    x_input = (torch.rand(batch_size, num_channels, num_time_bins) > 0.8).float()
    logits = model(x_input)
    
    assert logits.shape == (batch_size, num_classes)


def test_cochlea_profiler_metrics():
    """
    Nöromorfik Koklea Profilleyici Veri Sıkıştırma Kazancı Testi.
    """
    metrics = CochleaProfilleyici.profille(
        total_events=300,
        pcm_bytes=32000,
        snn_accuracy=96.5,
        latency_ms=1.5
    )
    
    assert metrics["compression_ratio_x"] > 1.0
    assert metrics["cochlea_readiness_score"] > 80.0
