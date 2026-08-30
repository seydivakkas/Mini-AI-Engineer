"""
Day 358: Deep Space Optical Communications & AI-Driven Adaptive Optics Wavefront Correction
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

from src.adaptive_optics_dsoc_motoru import (
    AtmosphericTurbulencePhaseScreen,
    DeformableMirrorController,
    DeepSpaceOpticalCommsSimulator,
    AdaptiveOpticsAIEngine,
)
from src.optics_profilleyici import OpticsProfilleyici


def test_atmospheric_turbulence_phase_screen():
    """
    Atmosferik Türbülans Faz Ekranı Testi.
    """
    sim = AtmosphericTurbulencePhaseScreen(grid_size=32)
    phase = sim.generate_turbulent_wavefront()
    assert phase.shape == (32, 32)
    assert not np.isnan(phase).any()


def test_deformable_mirror_controller_surface():
    """
    Deforme Ayna Yüzey Enterpolasyon Testi.
    """
    dm = DeformableMirrorController(grid_size=32, num_actuators_side=4)
    voltages = np.ones((4, 4)) * 2.0
    surface = dm.compute_dm_surface(voltages)
    assert surface.shape == (32, 32)


def test_strehl_ratio_calculation():
    """
    Strehl Oranı ve PSF Hesaplama Testi.
    """
    flat_phase = np.zeros((32, 32))
    strehl, psf = DeepSpaceOpticalCommsSimulator.compute_strehl_and_psf(flat_phase)
    assert strehl == pytest.approx(1.0, abs=0.05)
    assert psf.shape == (32, 32)


def test_optics_profiler_metrics():
    """
    Optik Profilleyici Metrik Testi.
    """
    mock_res = {
        "init_strehl": 0.04,
        "final_strehl": 0.88
    }
    metrics = OpticsProfilleyici.profille(mock_res)
    assert metrics["final_strehl"] == 0.88
    assert metrics["strehl_score"] == 88.0
    assert metrics["dsoc_readiness"] > 88.0
