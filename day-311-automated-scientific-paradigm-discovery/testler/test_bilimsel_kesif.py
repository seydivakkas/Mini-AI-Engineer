"""
Unit Tests for Day 311: Automated Scientific Theory Discovery
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

import sys
import os
import pytest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.bilimsel_kesif_motoru import (
    ScientificDiscoveryConfig,
    ScientificDiscoveryResult,
    CandidateLibrary,
    SINDyEquationDiscoverer,
    AutomatedScientificDiscoveryEngine
)
from src.bilimsel_kesif_profilleyici import ScientificDiscoveryProfiler
from src.gorsellestirici import ScientificDiscoveryGorsellestirici


@pytest.fixture
def base_config():
    return ScientificDiscoveryConfig(
        poly_order=3,
        include_trig=True,
        sparsity_threshold=0.08,
        ridge_alpha=1e-4,
        noise_level=0.005,
        time_steps=300,
        dt=0.02,
        seed=42
    )


def test_1_candidate_library_polynomial_generation():
    """Test 1: CandidateLibrary generates correct polynomial terms for 3D state."""
    lib = CandidateLibrary(poly_order=2, include_trig=False)
    X = np.random.randn(20, 3)
    Theta, names = lib.fit_transform(X)
    
    # 1 + 3 (linear) + 6 (quadratic) = 10 terms
    assert Theta.shape == (20, 10)
    assert "x1x2" in names
    assert "x3^2" in names


def test_2_candidate_library_trig_generation():
    """Test 2: CandidateLibrary includes sin and cos features when requested."""
    lib = CandidateLibrary(poly_order=1, include_trig=True)
    X = np.random.randn(20, 2)
    Theta, names = lib.fit_transform(X)
    
    # 1 + 2 (linear) + 4 (sin/cos) = 7 terms
    assert Theta.shape == (20, 7)
    assert "sin(x1)" in names
    assert "cos(x2)" in names


def test_3_sindy_stlsq_simple_linear():
    """Test 3: SINDy STLSQ recovers exact linear relationship from clean data."""
    discoverer = SINDyEquationDiscoverer(threshold=0.05, alpha=1e-4)
    Theta = np.random.randn(50, 4)
    # Target: dX = 2.5 * Theta[:, 1] - 1.2 * Theta[:, 3]
    dX = (2.5 * Theta[:, 1] - 1.2 * Theta[:, 3]).reshape(-1, 1)
    
    Xi = discoverer.fit(Theta, dX)
    assert abs(Xi[1, 0] - 2.5) < 0.05
    assert abs(Xi[3, 0] - (-1.2)) < 0.05
    assert abs(Xi[0, 0]) < 1e-4
    assert abs(Xi[2, 0]) < 1e-4


def test_4_sindy_stlsq_thresholding_effect():
    """Test 4: Coefficients smaller than threshold are zeroed out."""
    discoverer = SINDyEquationDiscoverer(threshold=0.2, alpha=1e-4)
    Theta = np.eye(5)
    dX = np.array([[0.05], [0.8], [0.02], [1.5], [0.01]])
    
    Xi = discoverer.fit(Theta, dX)
    assert Xi[0, 0] == 0.0
    assert Xi[1, 0] > 0.5
    assert Xi[2, 0] == 0.0
    assert Xi[3, 0] > 1.0


def test_5_format_equations_string():
    """Test 5: SINDy formats sparse matrix into readable differential equations."""
    discoverer = SINDyEquationDiscoverer()
    discoverer.Xi = np.array([
        [0.0, 0.0],
        [-10.0, 28.0],
        [10.0, -1.0]
    ])
    names = ["1", "x1", "x2"]
    eqs = discoverer.format_equations(names)
    
    assert "dx1/dt" in eqs
    assert "-10.000*x1" in eqs["dx1/dt"]
    assert "+10.000*x2" in eqs["dx1/dt"]


def test_6_lorenz_ode_and_data_generation(base_config):
    """Test 6: Dynamical trajectory generator produces valid states and derivatives."""
    engine = AutomatedScientificDiscoveryEngine(base_config)
    t, X, dX = engine.generate_data("lorenz")
    
    assert X.shape == (base_config.time_steps, 3)
    assert dX.shape == (base_config.time_steps, 3)
    assert len(t) == base_config.time_steps


def test_7_full_scientific_discovery_result(base_config):
    """Test 7: Autonomous discovery engine discovers true physical laws."""
    engine = AutomatedScientificDiscoveryEngine(base_config)
    res = engine.discover_laws()
    
    assert isinstance(res, ScientificDiscoveryResult)
    assert res.equation_recovery_precision_pct >= 85.0
    assert res.avg_parameter_relative_error_pct < 20.0
    assert len(res.discovered_equations) == 3


def test_8_profiler_and_dashboard_generation(base_config, tmp_path):
    """Test 8: Profiler extracts metrics and visualizer saves valid PNG dashboard."""
    engine = AutomatedScientificDiscoveryEngine(base_config)
    res = engine.discover_laws()
    
    prof = ScientificDiscoveryProfiler.profile_results(res)
    assert "equation_recovery_precision_pct" in prof
    assert "discovery_tier" in prof
    
    out_png = str(tmp_path / "test_bilimsel_paneli.png")
    ScientificDiscoveryGorsellestirici.ciz(res, out_png, prof)
    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1000
