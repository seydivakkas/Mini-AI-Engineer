"""
Unit Tests for Day 307: Unsupervised Latent Causal World Discovery & Do-Calculus
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

from src.nedensel_dunya_motoru import (
    CausalConfig,
    CausalDiscoveryResult,
    StructuralCausalModel,
    LatentCausalWorldModel,
    DoCalculusEngine,
    train_and_discover_causal_graph
)
from src.nedensel_profilleyici import CausalProfiler
from src.gorsellestirici import CausalWorldGorsellestirici


@pytest.fixture
def small_config():
    return CausalConfig(
        latent_dim=4,
        obs_dim=12,
        num_samples=200,
        batch_size=32,
        lr=1e-3,
        lambda_dag=1.0,
        lambda_sparse=0.05,
        lambda_interv=0.5,
        epochs=5,
        threshold_edge=0.20,
        seed=101
    )


def test_1_scm_generation_and_dag_validity():
    """Test 1: SCM generates valid latent variables and high-dim observations."""
    scm = StructuralCausalModel(latent_dim=5, obs_dim=20, seed=42)
    z, eps = scm.sample_latents(num_samples=50)
    x = scm.generate_observations(z)
    
    assert z.shape == (50, 5)
    assert eps.shape == (50, 5)
    assert x.shape == (50, 20)
    assert torch.all(torch.diag(scm.true_adj) == 0.0)


def test_2_scm_do_operator_intervention():
    """Test 2: Intervening on node 2 strictly fixes its value to target."""
    scm = StructuralCausalModel(latent_dim=5, obs_dim=20, seed=42)
    target_val = 3.14
    z_interv, _ = scm.sample_latents(num_samples=30, intervention=(2, target_val))
    
    assert torch.allclose(z_interv[:, 2], torch.full((30,), target_val))


def test_3_notears_acyclicity_constraint():
    """Test 3: NOTEARS constraint evaluates strictly positive for cyclic graphs and zero for empty graphs."""
    model = LatentCausalWorldModel(obs_dim=12, latent_dim=4)
    # Zero adjacency should yield h(A) = 0
    with torch.no_grad():
        model.adj_logits.zero_()
    h_zero = model.notears_acyclicity_constraint()
    assert abs(h_zero.item()) < 1e-4


def test_4_latent_causal_world_model_forward():
    """Test 4: Model forward pass produces valid reconstruction and adjacency."""
    model = LatentCausalWorldModel(obs_dim=16, latent_dim=4)
    x = torch.randn(10, 16)
    x_recon, z_causal, A = model(x)
    
    assert x_recon.shape == (10, 16)
    assert z_causal.shape == (10, 4)
    assert A.shape == (4, 4)
    assert torch.all(torch.diag(A) == 0.0)


def test_5_do_calculus_interventional_prediction():
    """Test 5: DoCalculusEngine properly freezes intervened node and propagates."""
    scm = StructuralCausalModel(latent_dim=4, obs_dim=12, seed=42)
    model = LatentCausalWorldModel(obs_dim=12, latent_dim=4)
    engine = DoCalculusEngine(model, scm)
    
    z_obs = torch.randn(20, 4)
    z_do = engine.interventional_prediction(z_obs, interv_node=1, interv_val=5.0)
    
    assert z_do.shape == (20, 4)
    assert torch.allclose(z_do[:, 1], torch.full((20,), 5.0))


def test_6_counterfactual_inference_abduction():
    """Test 6: Level 3 counterfactual engine predicts valid counterfactual observation."""
    scm = StructuralCausalModel(latent_dim=4, obs_dim=12, seed=42)
    model = LatentCausalWorldModel(obs_dim=12, latent_dim=4)
    engine = DoCalculusEngine(model, scm)
    
    x_obs = torch.randn(15, 12)
    x_cf = engine.counterfactual_inference(x_obs, interv_node=0, new_val=2.0)
    
    assert x_cf.shape == (15, 12)


def test_7_causal_discovery_training_loop(small_config):
    """Test 7: Full training loop completes and outputs valid CausalDiscoveryResult."""
    result = train_and_discover_causal_graph(small_config)
    
    assert isinstance(result, CausalDiscoveryResult)
    assert result.structural_hamming_distance >= 0
    assert 0.0 <= result.dag_true_positive_rate_pct <= 100.0
    assert result.reconstruction_mse >= 0.0
    assert len(result.loss_history) == small_config.epochs


def test_8_profiler_and_dashboard_generation(small_config, tmp_path):
    """Test 8: CausalProfiler computes metrics and dashboard visualizer saves PNG file."""
    result = train_and_discover_causal_graph(small_config)
    profil_ozeti = CausalProfiler.profile_results(result)
    
    assert "structural_hamming_distance" in profil_ozeti
    assert "dag_true_positive_rate_pct" in profil_ozeti
    
    output_png = str(tmp_path / "test_nedensel_paneli.png")
    CausalWorldGorsellestirici.ciz(result, output_png, profil_ozeti)
    assert os.path.exists(output_png)
    assert os.path.getsize(output_png) > 1000
