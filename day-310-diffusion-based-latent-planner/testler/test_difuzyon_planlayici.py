"""
Unit Tests for Day 310: Diffusion-Based Latent Planner
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

from src.difuzyon_planlayici_motoru import (
    DiffusionPlannerConfig,
    DiffusionPlannerResult,
    NoiseScheduler,
    SinusoidalPosEmb,
    TrajectoryUNet1D,
    GoalConditionedDiffusionPlanner
)
from src.difuzyon_profilleyici import DiffusionPlannerProfiler
from src.gorsellestirici import DiffusionPlannerGorsellestirici


@pytest.fixture
def base_config():
    return DiffusionPlannerConfig(
        trajectory_len=16,
        state_dim=4,
        num_diffusion_steps=10,
        guidance_scale=2.0,
        num_eval_trajectories=10,
        learning_rate=1e-3,
        seed=42
    )


def test_1_noise_scheduler_betas_and_alphas():
    """Test 1: Noise scheduler correctly computes decreasing alpha bar schedule."""
    sched = NoiseScheduler(num_timesteps=20)
    assert len(sched.betas) == 20
    assert torch.all(sched.alphas_cumprod[:-1] >= sched.alphas_cumprod[1:])
    assert 0.0 < sched.alphas_cumprod[-1].item() < 1.0


def test_2_noise_scheduler_q_sample_forward():
    """Test 2: Forward q_sample retains tensor shape and adds stochastic perturbation."""
    sched = NoiseScheduler(num_timesteps=20)
    x0 = torch.zeros(4, 16, 4)
    t = torch.tensor([5, 10, 15, 19])
    noisy = sched.q_sample(x0, t)
    
    assert noisy.shape == (4, 16, 4)
    assert not torch.allclose(noisy, x0)


def test_3_trajectory_unet1d_architecture():
    """Test 3: UNet1D accepts trajectory and conditions, outputting matching noise dimension."""
    net = TrajectoryUNet1D(state_dim=4, hidden_dim=32)
    x = torch.randn(2, 16, 4)
    t = torch.tensor([3, 7])
    goal = torch.tensor([[5.0, 5.0], [8.0, 8.0]])
    
    out = net(x, t, goal)
    assert out.shape == (2, 16, 4)


def test_4_sinusoidal_pos_emb():
    """Test 4: Sinusoidal embeddings produce expected embedding dimension."""
    emb = SinusoidalPosEmb(dim=32)
    t = torch.tensor([0, 10, 20])
    out = emb(t)
    assert out.shape == (3, 32)


def test_5_ddpm_reverse_sampling(base_config):
    """Test 5: DDPM reverse sampling generates smooth trajectories satisfying boundary start."""
    planner = GoalConditionedDiffusionPlanner(base_config)
    starts = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    goals = torch.tensor([[8.0, 8.0], [9.0, 9.0]])
    
    trajs = planner.sample_trajectories(starts, goals, use_ddim=False)
    assert trajs.shape == (2, base_config.trajectory_len, base_config.state_dim)
    assert torch.allclose(trajs[:, 0, :2], starts, atol=1e-4)


def test_6_ddim_accelerated_sampling(base_config):
    """Test 6: DDIM accelerated sampling runs successfully in reduced steps."""
    planner = GoalConditionedDiffusionPlanner(base_config)
    starts = torch.tensor([[1.0, 1.0]])
    goals = torch.tensor([[8.0, 8.0]])
    
    trajs = planner.sample_trajectories(starts, goals, use_ddim=True, ddim_steps=4)
    assert trajs.shape == (1, base_config.trajectory_len, base_config.state_dim)
    assert torch.allclose(trajs[:, 0, :2], starts, atol=1e-4)


def test_7_full_benchmark_evaluation(base_config):
    """Test 7: Full planner benchmark returns comprehensive metrics."""
    planner = GoalConditionedDiffusionPlanner(base_config)
    res = planner.evaluate_benchmark()
    
    assert isinstance(res, DiffusionPlannerResult)
    assert 0.0 <= res.goal_reachability_rate_pct <= 100.0
    assert 0.0 <= res.obstacle_avoidance_rate_pct <= 100.0
    assert res.sampled_trajectories.shape == (base_config.num_eval_trajectories, base_config.trajectory_len, 2)


def test_8_profiler_and_dashboard_generation(base_config, tmp_path):
    """Test 8: Profiler extracts metrics and visualizer saves valid PNG dashboard."""
    planner = GoalConditionedDiffusionPlanner(base_config)
    res = planner.evaluate_benchmark()
    
    prof = DiffusionPlannerProfiler.profile_results(res)
    assert "goal_reachability_rate_pct" in prof
    assert "planner_tier" in prof
    
    out_png = str(tmp_path / "test_difuzyon_paneli.png")
    DiffusionPlannerGorsellestirici.ciz(res, out_png, prof)
    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1000
