"""
Unit Tests for Day 314: Game-Theoretic Mechanism Design & Nash Bargaining
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

from src.oyun_teorisi_mekanizma import (
    MechanismConfig,
    MechanismResult,
    BargainingAgent,
    VCGMechanism,
    NashBargainingOptimizer,
    GameTheoreticEngine
)
from src.oyun_teorisi_profilleyici import GameTheoreticProfiler
from src.gorsellestirici import GameTheoreticGorsellestirici


@pytest.fixture
def base_config():
    return MechanismConfig(
        num_agents=3,
        num_goods_or_outcomes=4,
        total_compute_resource=80.0,
        seed=42
    )


def test_1_mechanism_config_initialization():
    """Test 1: Config initializes with correct hyperparameters."""
    cfg = MechanismConfig(num_agents=5, num_goods_or_outcomes=6, total_compute_resource=150.0)
    assert cfg.num_agents == 5
    assert cfg.num_goods_or_outcomes == 6
    assert cfg.total_compute_resource == 150.0


def test_2_bargaining_agent_initialization():
    """Test 2: Agent holds valid valuation and threat point data."""
    vals = np.array([10.0, 25.0, 5.0])
    agent = BargainingAgent(1, "Agent-1", vals, threat_point=4.0, bargaining_power=0.33, utility_weight=3.0)
    assert agent.agent_id == 1
    assert len(agent.valuations) == 3
    assert agent.threat_point == 4.0


def test_3_vcg_social_welfare_maximization():
    """Test 3: VCG selects the exact argmax outcome of sum(v_i)."""
    a1 = BargainingAgent(1, "A1", np.array([10.0, 40.0, 20.0]), 2.0, 0.5, 3.0)
    a2 = BargainingAgent(2, "A2", np.array([30.0, 20.0, 10.0]), 2.0, 0.5, 3.0)
    
    # Total welfares: [40, 60, 30] -> max at k=1 (60.0)
    k_star, welfare, pays, net_u = VCGMechanism.solve_allocation([a1, a2])
    assert k_star == 1
    assert welfare == 60.0


def test_4_vcg_positive_externality_payments():
    """Test 4: VCG payments are non-negative externalities."""
    a1 = BargainingAgent(1, "A1", np.array([10.0, 50.0]), 2.0, 0.5, 3.0)
    a2 = BargainingAgent(2, "A2", np.array([40.0, 10.0]), 2.0, 0.5, 3.0)
    
    _, _, pays, _ = VCGMechanism.solve_allocation([a1, a2])
    for p in pays.values():
        assert p >= 0.0


def test_5_vcg_dsic_truthful_dominance(base_config):
    """Test 5: Truthful bidding produces higher or equal net utility vs lying."""
    engine = GameTheoreticEngine(base_config)
    res = engine.run_simulation()
    assert res.truthful_vs_manipulated_utility_gain >= 0.0


def test_6_nash_bargaining_capacity_constraint(base_config):
    """Test 6: Total allocated resources match exact capacity constraint."""
    allocs, surpluses, prod = NashBargainingOptimizer.solve_bargaining(
        GameTheoreticEngine(base_config).agents, total_capacity=base_config.total_compute_resource
    )
    total_alloc = sum(allocs.values())
    assert np.isclose(total_alloc, base_config.total_compute_resource, atol=1e-2)


def test_7_nash_bargaining_individual_rationality(base_config):
    """Test 7: Every agent obtains positive surplus above threat point d_i."""
    engine = GameTheoreticEngine(base_config)
    res = engine.run_simulation()
    for s in res.nash_net_surpluses.values():
        assert s > 0.0 # Positive surplus


def test_8_profiler_and_dashboard_generation(base_config, tmp_path):
    """Test 8: Profiler produces summary metrics and dashboard generates PNG file."""
    engine = GameTheoreticEngine(base_config)
    res = engine.run_simulation()
    
    prof = GameTheoreticProfiler.profile_results(res)
    assert "vcg_social_welfare" in prof
    assert "pareto_efficiency_pct" in prof
    
    out_png = str(tmp_path / "test_oyun_paneli.png")
    GameTheoreticGorsellestirici.ciz(res, out_png, prof)
    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1000
