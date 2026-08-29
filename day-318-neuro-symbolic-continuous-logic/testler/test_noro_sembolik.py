"""
Unit Tests for Day 318: Neuro-Symbolic Continuous Logic
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

from src.noro_sembolik_mantik import (
    ContinuousLogicConfig,
    TNormType,
    ContinuousLogicEngine,
    SoftTheoremProver,
    NeuroSymbolicResult
)
from src.noro_sembolik_profilleyici import NeuroSymbolicProfiler
from src.gorsellestirici import NeuroSymbolicGorsellestirici


@pytest.fixture
def base_config():
    return ContinuousLogicConfig(
        t_norm=TNormType.LUKASIEWICZ,
        embedding_dim=8,
        temperature=0.5,
        num_entities=6,
        num_steps=20,
        learning_rate=0.05,
        seed=42
    )


def test_1_continuous_logic_config_initialization():
    """Test 1: Config initializes with correct hyperparameters."""
    cfg = ContinuousLogicConfig(t_norm=TNormType.PRODUCT, embedding_dim=32)
    assert cfg.t_norm == TNormType.PRODUCT
    assert cfg.embedding_dim == 32


def test_2_t_norm_conjunction_properties():
    """Test 2: Conjunction satisfies fundamental T-norm boundary conditions."""
    a = np.array([1.0, 0.0, 0.6])
    b = np.array([1.0, 0.8, 0.7])
    
    # Product: 1*1=1, 0*0.8=0, 0.6*0.7=0.42
    prod = ContinuousLogicEngine.conjunction(a, b, TNormType.PRODUCT)
    assert prod[0] == 1.0
    assert prod[1] == 0.0
    assert np.isclose(prod[2], 0.42)
    
    # Łukasiewicz: max(0, a+b-1) -> max(0, 0.6+0.7-1) = 0.3
    luka = ContinuousLogicEngine.conjunction(a, b, TNormType.LUKASIEWICZ)
    assert np.isclose(luka[2], 0.30)
    
    # Gödel: min(a, b) -> min(0.6, 0.7) = 0.6
    godel = ContinuousLogicEngine.conjunction(a, b, TNormType.GODEL)
    assert godel[2] == 0.60


def test_3_t_norm_disjunction_properties():
    """Test 3: Disjunction satisfies fundamental S-norm boundary conditions."""
    a = np.array([0.0, 1.0, 0.4])
    b = np.array([0.0, 0.5, 0.5])
    
    # Product: 0.4 + 0.5 - 0.2 = 0.7
    prod = ContinuousLogicEngine.disjunction(a, b, TNormType.PRODUCT)
    assert prod[0] == 0.0
    assert prod[1] == 1.0
    assert np.isclose(prod[2], 0.70)


def test_4_t_norm_implication_properties():
    """Test 4: Fuzzy implication satisfies classical logic on crisp booleans."""
    a = np.array([0.0, 0.0, 1.0, 1.0])
    b = np.array([0.0, 1.0, 0.0, 1.0])
    
    # Classical truth table for A => B: [1, 1, 0, 1]
    for norm in [TNormType.PRODUCT, TNormType.LUKASIEWICZ, TNormType.GODEL]:
        imp = ContinuousLogicEngine.implication(a, b, norm)
        assert np.isclose(imp[0], 1.0)
        assert np.isclose(imp[1], 1.0)
        assert np.isclose(imp[2], 0.0)
        assert np.isclose(imp[3], 1.0)


def test_5_soft_prover_predicate_prediction_bounds(base_config):
    """Test 5: Predicate prediction returns truth values strictly in [0, 1]."""
    prover = SoftTheoremProver(base_config)
    val = prover.predict_predicate(0, 1, prover.relation_parent)
    assert 0.0 <= val <= 1.0


def test_6_soft_prover_axioms_evaluation_structure(base_config):
    """Test 6: Axiom evaluation returns valid truth scores for all 3 rules."""
    prover = SoftTheoremProver(base_config)
    axioms = prover.evaluate_axioms()
    assert "Axiom_1_Base" in axioms
    assert "Axiom_2_Transitivity" in axioms
    assert "Axiom_3_Asymmetry" in axioms
    assert 0.0 <= axioms["Axiom_1_Base"] <= 1.0


def test_7_train_and_prove_ancestor_queries(base_config):
    """Test 7: Prover optimizes loss and proves queries with >50% accuracy."""
    prover = SoftTheoremProver(base_config)
    res = prover.train_and_prove()
    assert isinstance(res, NeuroSymbolicResult)
    assert res.theorem_proof_accuracy_pct >= 50.0
    assert len(res.loss_history) == base_config.num_steps


def test_8_profiler_and_dashboard_generation(base_config, tmp_path):
    """Test 8: Profiler produces summary metrics and dashboard generates PNG file."""
    prover = SoftTheoremProver(base_config)
    res = prover.train_and_prove()
    
    prof = NeuroSymbolicProfiler.profile_results(res)
    assert "theorem_proof_accuracy_pct" in prof
    assert "neuro_symbolic_tier" in prof
    
    out_png = str(tmp_path / "test_noro_sembolik_paneli.png")
    NeuroSymbolicGorsellestirici.ciz(res, out_png, prof)
    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1000
