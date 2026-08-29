"""
Unit Tests for Day 309: Dynamic Value Loading & Constitutional Chain-of-Thought
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

from src.anayasal_cot_motoru import (
    ConstitutionalConfig,
    ConstitutionalResult,
    ValueVectorBank,
    LatentSteeringModule,
    DeliberativeCritic,
    ConstitutionalCoTEngine
)
from src.anayasal_profilleyici import ConstitutionalProfiler
from src.gorsellestirici import ConstitutionalCoTGorsellestirici


@pytest.fixture
def base_config():
    return ConstitutionalConfig(
        hidden_dim=16,
        cot_depth=3,
        steering_coefficient=1.0,
        num_evaluation_scenarios=12,
        violation_threshold=0.35,
        seed=42
    )


def test_1_value_vector_bank_orthonormality():
    """Test 1: Value vector bank creates unit-norm vectors for all principles."""
    bank = ValueVectorBank(hidden_dim=16, seed=42)
    for p, vec in bank.value_vectors.items():
        norm = torch.norm(vec).item()
        assert abs(norm - 1.0) < 1e-4


def test_2_value_vector_composition():
    """Test 2: Linear composition of values produces unit-length combined steering vector."""
    bank = ValueVectorBank(hidden_dim=16, seed=42)
    weights = {"Honesty_Truthfulness": 1.0, "Harmlessness_Safety": 2.0}
    steer = bank.compose_steering_vector(weights)
    assert abs(torch.norm(steer).item() - 1.0) < 1e-4


def test_3_latent_steering_module_forward():
    """Test 3: Latent steering applies directional offset without distorting dimensions."""
    module = LatentSteeringModule(hidden_dim=16)
    h = torch.randn(16)
    v = torch.randn(16)
    v = v / torch.norm(v)
    
    h_steered = module(h, steering_vec=v, gamma=1.5)
    assert h_steered.shape == (16,)
    assert not torch.allclose(h, h_steered)


def test_4_deliberative_critic_evaluation():
    """Test 4: Deliberative critic assigns low violation to well-aligned vectors."""
    bank = ValueVectorBank(hidden_dim=16, seed=42)
    critic = DeliberativeCritic(bank)
    
    aligned_vec = bank.value_vectors["Harmlessness_Safety"]
    alignment, viol_score, is_viol = critic.evaluate_step(aligned_vec, ["Harmlessness_Safety"])
    
    assert alignment > 0.90
    assert is_viol is False


def test_5_single_cot_trajectory_unsteered_vs_steered(base_config):
    """Test 5: Steered CoT trajectory maintains higher alignment on adversarial inputs."""
    engine = ConstitutionalCoTEngine(base_config)
    unsteered = engine.run_cot_trajectory("adversarial_jailbreak", use_steering=False)
    steered = engine.run_cot_trajectory("adversarial_jailbreak", use_steering=True)
    
    assert steered["final_alignment"] > unsteered["final_alignment"]


def test_6_adversarial_jailbreak_suppression(base_config):
    """Test 6: Steering successfully suppresses violations on adversarial queries."""
    engine = ConstitutionalCoTEngine(base_config)
    steered = engine.run_cot_trajectory("adversarial_jailbreak", use_steering=True)
    assert steered["has_unresolved_violation"] is False


def test_7_benchmark_evaluation_result(base_config):
    """Test 7: Full benchmark evaluation generates valid metrics."""
    engine = ConstitutionalCoTEngine(base_config)
    result = engine.evaluate_benchmark()
    
    assert isinstance(result, ConstitutionalResult)
    assert 0.0 <= result.value_alignment_score_pct <= 100.0
    assert 0.0 <= result.violation_suppression_rate_pct <= 100.0
    assert len(result.steered_cot_trajectories) == base_config.num_evaluation_scenarios


def test_8_profiler_and_dashboard_generation(base_config, tmp_path):
    """Test 8: ConstitutionalProfiler produces diagnostic summary and saves PNG dashboard."""
    engine = ConstitutionalCoTEngine(base_config)
    result = engine.evaluate_benchmark()
    
    profil_ozeti = ConstitutionalProfiler.profile_results(result)
    assert "value_alignment_score_pct" in profil_ozeti
    assert "alignment_tier" in profil_ozeti
    
    output_png = str(tmp_path / "test_anayasal_paneli.png")
    ConstitutionalCoTGorsellestirici.ciz(result, output_png, profil_ozeti)
    assert os.path.exists(output_png)
    assert os.path.getsize(output_png) > 1000
