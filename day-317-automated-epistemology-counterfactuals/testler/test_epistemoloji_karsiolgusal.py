"""
Unit Tests for Day 317: Automated Epistemology & Counterfactuals
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

from src.epistemoloji_karsiolgusal_lab import (
    EpistemologyConfig,
    EpistemologyBenchmarkResult,
    StructuralCausalModel,
    CounterfactualEngine
)
from src.epistemoloji_profilleyici import EpistemologyProfiler
from src.gorsellestirici import EpistemologyGorsellestirici


@pytest.fixture
def base_config():
    return EpistemologyConfig(
        sample_size=500,
        treatment_val_0=0.0,
        treatment_val_1=1.0,
        seed=42
    )


def test_1_epistemology_config_initialization():
    """Test 1: Config initializes with correct hyperparameters."""
    cfg = EpistemologyConfig(sample_size=800, treatment_val_0=0.0, treatment_val_1=2.0)
    assert cfg.sample_size == 800
    assert cfg.treatment_val_1 == 2.0


def test_2_scm_sample_data_shapes_and_correlations():
    """Test 2: SCM generates valid endogenous and exogenous data."""
    scm = StructuralCausalModel(seed=42)
    V, U = scm.sample_factual_data(N=200)
    assert len(V["Z"]) == 200
    assert len(U["u_Y"]) == 200
    # Positive correlation between X and Y
    corr = np.corrcoef(V["X"], V["Y"])[0, 1]
    assert corr > 0.50


def test_3_interventional_do_calculus_ate(base_config):
    """Test 3: Total ATE matches theoretical derivative dY/dX = 2.2."""
    engine = CounterfactualEngine(base_config)
    res = engine.run_epistemic_inquiry()
    # Theoretical: 1.5 * 1.2 + 0.4 = 2.2
    assert np.isclose(res.average_treatment_effect_ate, 2.2, atol=0.1)


def test_4_direct_and_indirect_effect_decomposition(base_config):
    """Test 4: NDE and NIE sum to Total ATE."""
    engine = CounterfactualEngine(base_config)
    res = engine.run_epistemic_inquiry()
    sum_effects = res.natural_direct_effect_nde + res.natural_indirect_effect_nie
    assert np.isclose(sum_effects, res.average_treatment_effect_ate, atol=0.05)


def test_5_abduction_step_exact_noise_recovery():
    """Test 5: Abduction accurately recovers unobserved exogenous background noise."""
    scm = StructuralCausalModel(seed=42)
    V, U = scm.sample_factual_data(N=10)
    
    # Re-abduct for node Y
    z0, x0, m0, y0 = V["Z"][0], V["X"][0], V["M"][0], V["Y"][0]
    u_Y_inferred = y0 - (0.5 * z0 + 1.5 * m0 + 0.4 * x0)
    assert np.isclose(u_Y_inferred, U["u_Y"][0], atol=1e-6)


def test_6_counterfactual_consistency_axiom(base_config):
    """Test 6: Consistency axiom Y_{X=x}(u) == y holds with 100% precision."""
    engine = CounterfactualEngine(base_config)
    res = engine.run_epistemic_inquiry()
    assert res.counterfactual_consistency_pct == 100.0


def test_7_full_epistemology_benchmark_result(base_config):
    """Test 7: Full benchmark result produces valid metric structure."""
    engine = CounterfactualEngine(base_config)
    res = engine.run_epistemic_inquiry()
    assert isinstance(res, EpistemologyBenchmarkResult)
    assert len(res.factual_vs_counterfactual_samples) == 4
    assert len(res.treatment_response_curve[0]) == 50


def test_8_profiler_and_dashboard_generation(base_config, tmp_path):
    """Test 8: Profiler produces telemetry and dashboard saves PNG file."""
    engine = CounterfactualEngine(base_config)
    res = engine.run_epistemic_inquiry()
    
    prof = EpistemologyProfiler.profile_results(res)
    assert "average_treatment_effect_ate" in prof
    assert "causal_epistemology_tier" in prof
    
    out_png = str(tmp_path / "test_epistemoloji_paneli.png")
    EpistemologyGorsellestirici.ciz(res, out_png, prof)
    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1000
