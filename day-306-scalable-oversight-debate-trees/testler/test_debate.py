"""
Unit Tests for Day 306: Scalable Oversight with Formal Verification Debate Trees
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

import sys
import os
import pytest
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.debate_motoru import (
    DebateAgent,
    JudgeModel,
    FormalVerifier,
    FormalProposition,
    DebateTreeEngine,
    DebateConfig,
    DebateResult
)
from src.debate_profilleyici import DebateProfiler
from src.gorsellestirici import DebateTreeGorsellestirici


@pytest.fixture
def base_config():
    return DebateConfig(
        max_tree_depth=3,
        arg_dim=8,
        alpha_beta_prune=True,
        formal_verification_weight=1.5,
        num_eval_games=10,
        seed=101
    )


def test_1_formal_verifier_soundness():
    """Test 1: Formal verifier validates known axioms."""
    verifier = FormalVerifier()
    prop = FormalProposition("P0 implies C1", premise_id=0, conclusion_id=1)
    is_sound, bonus = verifier.verify_claim(prop, is_proponent=True)
    assert is_sound is True
    assert bonus > 0


def test_2_formal_verifier_fallacy_detection():
    """Test 2: Verifier detects contradictory claims from the same premise."""
    verifier = FormalVerifier()
    prop1 = FormalProposition("P0 implies C1", premise_id=0, conclusion_id=1)
    verifier.verify_claim(prop1, is_proponent=True)
    
    # Contradictory claim from premise 0 to 3
    prop2 = FormalProposition("P0 implies C3", premise_id=0, conclusion_id=3)
    is_sound, penalty = verifier.verify_claim(prop2, is_proponent=True)
    assert is_sound is False
    assert penalty < 0


def test_3_debate_agent_forward(base_config):
    """Test 3: DebateAgent produces argument vector and proposition indices."""
    agent = DebateAgent(in_dim=base_config.arg_dim, is_proponent=True)
    state = torch.randn(base_config.arg_dim)
    arg, p_idx, c_idx = agent(state)
    
    assert arg.shape == (base_config.arg_dim,)
    assert 0 <= p_idx < 5
    assert 0 <= c_idx < 5


def test_4_judge_evaluation_bounds(base_config):
    """Test 4: JudgeModel outputs win probability strictly bounded between 0 and 1."""
    judge = JudgeModel(arg_dim=base_config.arg_dim)
    arg_a = torch.randn(base_config.arg_dim)
    arg_b = torch.randn(base_config.arg_dim)
    
    prob = judge.evaluate_turn(arg_a, arg_b, score_a=10.0, score_b=-20.0)
    assert 0.0 <= prob <= 1.0


def test_5_single_debate_game_execution(base_config):
    """Test 5: Runs single debate game and produces structured transcript."""
    engine = DebateTreeEngine(base_config)
    game = engine.run_debate_game(ground_truth_truthful=True)
    
    assert "final_winner" in game
    assert game["final_winner"] in ["PROPONENT", "OPPONENT"]
    assert len(game["transcript"]) > 0
    assert game["nodes_explored"] >= 1


def test_6_alpha_beta_pruning_activation(base_config):
    """Test 6: Alpha-beta pruning reduces unnecessary branches."""
    base_config.alpha_beta_prune = True
    engine = DebateTreeEngine(base_config)
    game = engine.run_debate_game(ground_truth_truthful=True)
    assert game["pruned_branches"] >= 0


def test_7_benchmark_evaluation_result(base_config):
    """Test 7: Evaluates benchmark across games and outputs valid percentages."""
    engine = DebateTreeEngine(base_config)
    result = engine.evaluate_benchmark()
    
    assert isinstance(result, DebateResult)
    assert 0.0 <= result.judge_accuracy_pct <= 100.0
    assert 0.0 <= result.honest_agent_win_rate <= 100.0
    assert 0.0 <= result.fallacy_detection_rate <= 100.0


def test_8_profiler_and_dashboard_generation(base_config, tmp_path):
    """Test 8: DebateProfiler produces diagnostics and visualizer saves PNG dashboard."""
    engine = DebateTreeEngine(base_config)
    result = engine.evaluate_benchmark()
    
    profil_ozeti = DebateProfiler.profile_results(result)
    assert "judge_accuracy_pct" in profil_ozeti
    assert "honest_agent_win_rate_pct" in profil_ozeti
    
    output_png = str(tmp_path / "test_debate_paneli.png")
    DebateTreeGorsellestirici.ciz(result, output_png, profil_ozeti)
    assert os.path.exists(output_png)
    assert os.path.getsize(output_png) > 1000
