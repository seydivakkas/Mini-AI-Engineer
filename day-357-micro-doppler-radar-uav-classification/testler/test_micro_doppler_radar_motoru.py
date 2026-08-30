"""
Day 357: Radar Micro-Doppler Signature Classification for Micro-UAVs and Ballistic Targets
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

from src.micro_doppler_radar_motoru import (
    RadarTargetType,
    MicroDopplerSignalSynthesizer,
    TimeFrequencySpectrogramEngine,
    MicroDopplerDeepClassifier,
    AirDefenseRadarTargetAnalyzer,
)
from src.radar_profilleyici import RadarProfilleyici


def test_micro_doppler_signal_synthesizer_shape():
    """
    Radar Sinyal Sentezleyici Çıkış Testi.
    """
    synth = MicroDopplerSignalSynthesizer(duration_sec=0.5)
    sig = synth.synthesize_target_signal(RadarTargetType.QUADCOPTER_DRONE)
    assert len(sig) == 2000
    assert np.iscomplexobj(sig)


def test_stft_spectrogram_engine():
    """
    STFT Spektrogram Boyut Testi.
    """
    synth = MicroDopplerSignalSynthesizer(duration_sec=0.5)
    sig = synth.synthesize_target_signal(RadarTargetType.BIRD_FLAPPING)
    stft_eng = TimeFrequencySpectrogramEngine()
    f, t, Sxx_dB = stft_eng.compute_spectrogram(sig)
    
    assert len(f) > 0
    assert len(t) > 0
    assert Sxx_dB.shape == (len(f), len(t))


def test_micro_doppler_deep_classifier_quadcopter():
    """
    Mikro-Doppler Sınıflandırıcı Testi.
    """
    analyzer = AirDefenseRadarTargetAnalyzer()
    res = analyzer.analyze_all_targets()
    assert res["accuracy_pct"] == 100.0


def test_radar_profiler_metrics():
    """
    Radar Profilleyici Metrik Testi.
    """
    mock_res = {
        "accuracy_pct": 100.0
    }
    metrics = RadarProfilleyici.profille(mock_res)
    assert metrics["accuracy_pct"] == 100.0
    assert metrics["radar_ai_readiness"] == 100.0
