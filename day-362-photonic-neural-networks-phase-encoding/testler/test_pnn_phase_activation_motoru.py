"""
Day 362: Photonic Neural Networks (PNN) with Phase Encoding & Electro-Optic Activations
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

from src.pnn_phase_activation_motoru import (
    OpticalPhaseEncoder,
    ElectroOpticActivationFunction,
    PhotonicLinearLayer,
    DeepPhotonicNeuralNetwork,
)
from src.pnn_profilleyici import PNNProfilleyici


def test_optical_phase_encoder_output():
    """
    Optik Faz Kodlayıcı Çıktı Testi.
    """
    encoder = OpticalPhaseEncoder(p_laser_mw=10.0)
    x = np.array([0.0, 0.5, -0.5, 1.0])
    e_field = encoder.encode(x)
    assert e_field.shape == (4,)
    assert np.iscomplexobj(e_field)
    np.testing.assert_allclose(np.abs(e_field)**2, 10.0, atol=1e-5)


def test_electro_optic_activation_nonlinearity():
    """
    Elektro-Optik Aktivasyon Testi.
    """
    act = ElectroOpticActivationFunction(i_sat=2.0)
    i_vals = np.array([0.0, 1.0, 2.0, 3.0])
    act_out = act.apply_activation(i_vals)
    assert act_out.shape == (4,)
    assert (act_out >= 0.0).all()


def test_photonic_linear_layer_forward():
    """
    Fotonik Lineer Katman Çıkarım Testi.
    """
    layer = PhotonicLinearLayer(in_dim=4, out_dim=8)
    e_in = np.ones(4)
    out_intensity = layer.forward_optical(e_in)
    assert out_intensity.shape == (8,)
    assert not np.isnan(out_intensity).any()


def test_pnn_profiler_metrics():
    """
    PNN Profilleyici Metrik Testi.
    """
    mock_res = {
        "accuracy": 98.0,
        "photonic_latency_ps": 43.2
    }
    metrics = PNNProfilleyici.profille(mock_res)
    assert metrics["accuracy"] == 98.0
    assert metrics["accuracy_score"] == 98.0
    assert metrics["deep_pnn_readiness"] > 98.0
