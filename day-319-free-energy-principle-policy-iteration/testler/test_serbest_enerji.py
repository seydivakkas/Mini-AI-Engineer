"""
Unit Tests for Day 319: Free Energy Principle & Active Inference
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

from src.serbest_enerji_aktif_cikarim import (
    FEPConfig,
    FEPSimulationResult,
    GenerativeEnvironment,
    ActiveInferenceAgent
)
from src.serbest_enerji_profilleyici import FEPProfiler
from src.gorsellestirici import FEPGorsellestirici


@pytest.fixture
def base_config():
    return FEPConfig(
        num_states=4,
        num_obs=4,
        num_actions=3,
        horizon=6,
        precision_gamma=8.0,
        seed=42
    )


def test_1_fep_config_initialization():
    """Test 1: Config initializes with correct hyperparameters."""
    cfg = FEPConfig(num_states=5, precision_gamma=12.0)
    assert cfg.num_states == 5
    assert cfg.precision_gamma == 12.0


def test_2_generative_environment_step():
    """Test 2: Environment executes transitions correctly."""
    env = GenerativeEnvironment(true_reward_target=2, seed=42)
    next_s, obs = env.step(action=0)
    assert next_s == 1
    assert obs == 1


def test_3_active_inference_matrices_shapes(base_config):
    """Test 3: Generative model matrices A, B, C, D have proper dimensionalities."""
    agent = ActiveInferenceAgent(base_config)
    assert agent.A_mat.shape == (base_config.num_obs, base_config.num_states)
    assert agent.B_mat.shape == (base_config.num_states, base_config.num_states, base_config.num_actions)
    assert len(agent.C_vec) == base_config.num_obs
    assert len(agent.D_vec) == base_config.num_states


def test_4_belief_updating_vfe_minimization(base_config):
    """Test 4: Belief updating normalizes probabilities and computes finite VFE."""
    agent = ActiveInferenceAgent(base_config)
    vfe = agent.update_beliefs(obs=1, prev_action=0)
    assert np.isclose(np.sum(agent.q_s), 1.0)
    assert np.isfinite(vfe)


def test_5_expected_free_energy_computation(base_config):
    """Test 5: EFE computes pragmatic and epistemic components for all actions."""
    agent = ActiveInferenceAgent(base_config)
    G, prag, epis = agent.evaluate_expected_free_energy()
    assert len(G) == base_config.num_actions
    assert np.all(np.isfinite(G))
    assert np.all(np.isfinite(epis))


def test_6_action_selection_with_precision(base_config):
    """Test 6: Action selection returns valid action within discrete domain."""
    agent = ActiveInferenceAgent(base_config)
    action, g_val, p_val, e_val = agent.select_action()
    assert 0 <= action < base_config.num_actions
    assert np.isfinite(g_val)


def test_7_full_active_inference_loop_trajectory(base_config):
    """Test 7: Simulation loop runs and records trajectory."""
    agent = ActiveInferenceAgent(base_config)
    env = GenerativeEnvironment(true_reward_target=2, seed=42)
    res = agent.run_active_inference_loop(env)
    assert isinstance(res, FEPSimulationResult)
    assert len(res.trajectory_actions) > 0
    assert len(res.variational_free_energy_history) > 0


def test_8_profiler_and_dashboard_generation(base_config, tmp_path):
    """Test 8: Profiler produces summary metrics and visualizer writes PNG file."""
    agent = ActiveInferenceAgent(base_config)
    env = GenerativeEnvironment(true_reward_target=2, seed=42)
    res = agent.run_active_inference_loop(env)
    
    prof = FEPProfiler.profile_results(res)
    assert "entropy_reduction_pct" in prof
    assert "fep_agent_tier" in prof
    
    out_png = str(tmp_path / "test_serbest_enerji_paneli.png")
    FEPGorsellestirici.ciz(res, agent, out_png, prof)
    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1000
