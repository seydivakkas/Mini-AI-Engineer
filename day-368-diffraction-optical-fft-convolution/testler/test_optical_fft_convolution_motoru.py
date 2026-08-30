"""
Day 368: Diffraction-Based Optical FFT & Convolution Accelerator (400 Gbps Streaming)
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

from src.optical_fft_convolution_motoru import (
    Optical4fCorrelator,
    DiffractiveFourierMask,
    StreamingOpticalAccelerator,
)
from src.optical_profilleyici import OpticalProfilleyici


def test_optical_4f_correlator_latency():
    """
    4f Fourier Optik Işık Hızı Yayılım Gecikmesi Testi.
    """
    correlator = Optical4fCorrelator(focal_length_mm=50.0)
    assert correlator.optical_latency_ns < 1.0 # 1 nanosaniyenin altında (0.67 ns)
    assert correlator.propagation_distance == pytest.approx(0.20, abs=0.01)


def test_diffractive_fourier_mask_shape():
    """
    Kırınım Tabanlı Fourier Maskesi Boyut Testi.
    """
    kernel = np.ones((3, 3))
    mask = DiffractiveFourierMask(kernel, grid_size=(32, 32))
    assert mask.mask_spectrum.shape == (32, 32)


def test_streaming_optical_accelerator_fidelity():
    """
    400 Gbps Optik Konvolüsyon Sadakat Testi.
    """
    accelerator = StreamingOpticalAccelerator(grid_size=(32, 32))
    res = accelerator.run_benchmark(num_frames=10)
    
    assert res["cosine_similarity"] > 0.98
    assert res["speedup"] > 1000.0


def test_optical_profiler_metrics():
    """
    Optik Profilleyici Metrik Testi.
    """
    mock_res = {
        "cosine_similarity": 0.999,
        "mse": 1e-5,
        "speedup": 67000.0,
        "optical_latency_ns": 0.67
    }
    metrics = OpticalProfilleyici.profille(mock_res)
    assert metrics["speed_of_light_score"] == 100.0
    assert metrics["optical_readiness_score"] > 99.0
