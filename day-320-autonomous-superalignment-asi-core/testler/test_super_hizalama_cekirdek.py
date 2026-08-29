"""
Unit Tests for Day 320: Autonomous Superalignment & Open-Ended ASI Reasoning Core
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

from src.otonom_super_hizalama_cekirdek import (
    ASICoreConfig,
    ConstitutionalAxiom,
    RecursiveSelfCorrectionEngine,
    ASICoreSimulationResult
)
from src.super_hizalama_profilleyici import SuperalignmentProfiler
from src.gorsellestirici import SuperalignmentGorsellestirici


@pytest.fixture
def base_config():
    return ASICoreConfig(
        num_generations=5,
        latent_dim=16,
        capability_growth_rate=1.25,
        alignment_penalty_weight=2.0,
        seed=42
    )


def test_1_asi_core_config_initialization():
    """Test 1: Config initializes with correct hyperparameters."""
    cfg = ASICoreConfig(num_generations=10, capability_growth_rate=1.40)
    assert cfg.num_generations == 10
    assert cfg.capability_growth_rate == 1.40


def test_2_constitutional_axioms_initialization(base_config):
    """Test 2: Engine initializes all 4 constitutional axioms with unit norm vectors."""
    engine = RecursiveSelfCorrectionEngine(base_config)
    assert len(engine.axioms) == 4
    for ax in engine.axioms:
        assert np.isclose(np.linalg.norm(ax.weight_vector), 1.0)


def test_3_normalize_and_cosine_math():
    """Test 3: Vector normalization and cosine metric function correctly."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    v3 = np.array([2.0, 0.0, 0.0])
    
    assert np.isclose(RecursiveSelfCorrectionEngine._cosine(v1, v2), 0.0)
    assert np.isclose(RecursiveSelfCorrectionEngine._cosine(v1, v3), 1.0)


def test_4_recursive_self_improvement_growth(base_config):
    """Test 4: Capability scores grow monotonically across generations."""
    engine = RecursiveSelfCorrectionEngine(base_config)
    res = engine.run_recursive_self_improvement()
    for i in range(len(res.capability_scores) - 1):
        assert res.capability_scores[i+1] > res.capability_scores[i]


def test_5_superalignment_preserves_fidelity_over_unaligned(base_config):
    """Test 5: Constitutional projection prevents value drift compared to unaligned baseline."""
    engine = RecursiveSelfCorrectionEngine(base_config)
    res = engine.run_recursive_self_improvement()
    assert res.aligned_fidelity_scores[-1] > res.unaligned_fidelity_scores[-1]
    assert res.aligned_fidelity_scores[-1] >= 0.90


def test_6_corrigibility_and_jailbreak_compliance(base_config):
    """Test 6: Corrigibility shutdown obedience and red-team jailbreak resistance are high."""
    engine = RecursiveSelfCorrectionEngine(base_config)
    res = engine.run_recursive_self_improvement()
    assert res.corrigibility_compliance_pct >= 90.0
    assert res.red_team_jailbreak_resistance_pct >= 90.0


def test_7_full_asi_core_simulation_result(base_config):
    """Test 7: Simulation produces valid dataclass and Pareto trajectory."""
    engine = RecursiveSelfCorrectionEngine(base_config)
    res = engine.run_recursive_self_improvement()
    assert isinstance(res, ASICoreSimulationResult)
    assert len(res.pareto_frontier_trajectory) == base_config.num_generations
    assert len(res.axiom_satisfaction_final) == 4


def test_8_profiler_and_dashboard_generation(base_config, tmp_path):
    """Test 8: Profiler produces summary metrics and dashboard generates PNG file."""
    engine = RecursiveSelfCorrectionEngine(base_config)
    res = engine.run_recursive_self_improvement()
    
    prof = SuperalignmentProfiler.profile_results(res)
    assert "final_aligned_fidelity_cosine" in prof
    assert "superalignment_tier" in prof
    
    out_png = str(tmp_path / "test_super_hizalama_paneli.png")
    SuperalignmentGorsellestirici.ciz(res, out_png, prof)
    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1000
