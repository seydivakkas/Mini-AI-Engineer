"""
Unit Tests for Day 316: Adversarial Byzantine Fault Tolerance
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

from src.bizans_hata_toleransi import (
    ByzantineSwarmConfig,
    ByzantineBenchmarkResult,
    ByzantineAggregatorBank,
    ByzantineDefenseEngine
)
from src.bizans_profilleyici import ByzantineDefenseProfiler
from src.gorsellestirici import ByzantineDefenseGorsellestirici


@pytest.fixture
def base_config():
    return ByzantineSwarmConfig(
        num_nodes=10,
        num_byzantine=2,
        param_dim=20,
        attack_type="sign_flipping",
        iterations=15,
        learning_rate=0.05,
        seed=42
    )


def test_1_config_initialization():
    """Test 1: Config initializes with valid Byzantine fraction."""
    cfg = ByzantineSwarmConfig(num_nodes=12, num_byzantine=3)
    assert cfg.num_nodes == 12
    assert cfg.num_byzantine == 3


def test_2_naive_mean_vulnerability_to_sign_flipping():
    """Test 2: Naive mean gets corrupted by amplified sign-flipping attackers."""
    honest = np.ones((6, 10)) * 2.0
    attackers = np.ones((3, 10)) * -10.0 # Extreme negative push
    pool = np.vstack([honest, attackers])
    
    mean_g = ByzantineAggregatorBank.naive_mean(pool)
    # (6 * 2 + 3 * -10) / 9 = -18 / 9 = -2.0 (Corrupted negative gradient)
    assert np.all(mean_g < 0.0)


def test_3_coordinate_median_aggregation():
    """Test 3: Coordinate median effectively ignores extreme scalar outliers."""
    honest = np.ones((7, 10)) * 2.0
    attackers = np.ones((2, 10)) * 1000.0
    pool = np.vstack([honest, attackers])
    
    med_g = ByzantineAggregatorBank.coordinate_median(pool)
    assert np.allclose(med_g, np.ones(10) * 2.0, atol=1e-3)


def test_4_trimmed_mean_aggregation():
    """Test 4: Trimmed mean discards top and bottom f outliers."""
    honest = np.array([[1.0], [2.0], [2.0], [2.0], [3.0]])
    attackers = np.array([[-50.0], [100.0]])
    pool = np.vstack([honest, attackers]) # 7 nodes, f=2
    
    tm_g = ByzantineAggregatorBank.trimmed_mean(pool, num_byzantine=2)
    assert np.isclose(tm_g[0], 2.0, atol=0.2)


def test_5_multi_krum_selection_and_aggregation():
    """Test 5: Multi-Krum chooses honest candidate nodes based on distance scoring."""
    honest = np.ones((6, 5)) * 1.0
    attackers = np.ones((2, 5)) * -50.0
    pool = np.vstack([honest, attackers])
    
    g_krum, selected = ByzantineAggregatorBank.multi_krum(pool, num_byzantine=2, m=3)
    assert len(selected) == 3
    # Selected should be from honest nodes (indices 0..5)
    for idx in selected:
        assert idx < 6
    assert np.allclose(g_krum, np.ones(5) * 1.0, atol=1e-3)


def test_6_bulyan_robustness():
    """Test 6: Bulyan combines Krum and Trimmed Mean to filter Byzantine attackers."""
    honest = np.ones((9, 10)) * 3.0
    attackers = np.ones((3, 10)) * -30.0
    pool = np.vstack([honest, attackers])
    
    bulyan_g = ByzantineAggregatorBank.bulyan(pool, num_byzantine=3)
    assert np.all(bulyan_g > 0.0) # Maintains positive honest gradient direction


def test_7_full_benchmark_result_structure(base_config):
    """Test 7: Full benchmark produces valid result dataclass with high mitigation."""
    engine = ByzantineDefenseEngine(base_config)
    res = engine.run_defense_benchmark()
    
    assert isinstance(res, ByzantineBenchmarkResult)
    assert res.attack_mitigation_ratio_pct > 70.0
    assert res.mean_cosine_fidelity["Bulyan"] > 0.80


def test_8_profiler_and_dashboard_generation(base_config, tmp_path):
    """Test 8: Profiler produces summary metrics and dashboard generates PNG file."""
    engine = ByzantineDefenseEngine(base_config)
    res = engine.run_defense_benchmark()
    
    prof = ByzantineDefenseProfiler.profile_results(res)
    assert "attack_mitigation_ratio_pct" in prof
    assert "resilience_tier" in prof
    
    out_png = str(tmp_path / "test_bizans_paneli.png")
    ByzantineDefenseGorsellestirici.ciz(res, out_png, prof)
    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1000
