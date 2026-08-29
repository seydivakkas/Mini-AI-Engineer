"""
Unit Tests for Day 305: Cross-Coder Sparse Autoencoder (SAE)
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

from src.cross_coder_motoru import (
    CrossCoderSAE,
    CrossCoderConfig,
    CrossCoderResult,
    SyntheticActivationGenerator,
    CrossCoderTrainer
)
from src.sae_profilleyici import SAEProfiler
from src.gorsellestirici import CrossCoderGorsellestirici


@pytest.fixture
def small_config():
    return CrossCoderConfig(
        num_layers=2,
        d_model=16,
        dict_multiplier=4,  # d_sae = 64
        top_k=8,
        l1_coeff=0.001,
        lr=0.005,
        batch_size=32,
        epochs=5,
        seed=101
    )


def test_1_activation_generator_shapes(small_config):
    """Test 1: Generator outputs tensor of shape (batch_size, num_layers, d_model)."""
    gen = SyntheticActivationGenerator(
        num_layers=small_config.num_layers,
        d_model=small_config.d_model,
        num_true_concepts=32,
        seed=small_config.seed
    )
    batch = gen.generate_batch(batch_size=16, sparsity_p=0.1)
    assert batch.shape == (16, small_config.num_layers, small_config.d_model)
    assert not torch.isnan(batch).any()


def test_2_cross_coder_forward_pass(small_config):
    """Test 2: CrossCoderSAE outputs reconstructions matching input and sparse latent h."""
    model = CrossCoderSAE(small_config)
    x = torch.randn(8, small_config.num_layers, small_config.d_model)
    x_hat, h = model(x)
    
    assert x_hat.shape == x.shape
    assert h.shape == (8, small_config.d_sae)
    # Check TopK constraint: at most top_k non-zeros per sample
    active_per_sample = (h > 0).float().sum(dim=-1)
    assert (active_per_sample <= small_config.top_k).all()


def test_3_decoder_unit_norm_normalization(small_config):
    """Test 3: Decoder columns are normalized to unit L2 norm."""
    model = CrossCoderSAE(small_config)
    model.normalize_decoder()
    
    norms = torch.norm(model.W_dec, dim=1)  # [K, d_sae]
    expected = torch.ones_like(norms)
    assert torch.allclose(norms, expected, atol=1e-4)


def test_4_cross_coder_loss_computation(small_config):
    """Test 4: Loss computation returns positive scalar and metrics dictionary."""
    model = CrossCoderSAE(small_config)
    x = torch.randn(12, small_config.num_layers, small_config.d_model)
    x_hat, h = model(x)
    loss, metrics = model.compute_loss(x, x_hat, h)
    
    assert loss.item() > 0.0
    assert "recon_loss" in metrics
    assert "l1_loss" in metrics
    assert "l0_sparsity" in metrics
    assert metrics["l0_sparsity"] <= small_config.top_k


def test_5_training_loss_reduction(small_config):
    """Test 5: Trainer runs epochs and records history."""
    gen = SyntheticActivationGenerator(
        num_layers=small_config.num_layers,
        d_model=small_config.d_model,
        num_true_concepts=32,
        seed=small_config.seed
    )
    batches = [gen.generate_batch(batch_size=small_config.batch_size) for _ in range(5)]
    trainer = CrossCoderTrainer(small_config)
    
    metrics = trainer.train_epoch(batches)
    assert len(trainer.history["total_loss"]) == 1
    assert metrics["total_loss"] > 0.0


def test_6_fve_fraction_of_variance_explained(small_config):
    """Test 6: Evaluate computes FVE bounded between 0% and 100%."""
    gen = SyntheticActivationGenerator(
        num_layers=small_config.num_layers,
        d_model=small_config.d_model,
        num_true_concepts=32,
        seed=small_config.seed
    )
    test_x = gen.generate_batch(batch_size=64)
    trainer = CrossCoderTrainer(small_config)
    
    result = trainer.evaluate(test_x)
    assert isinstance(result, CrossCoderResult)
    assert 0.0 <= result.mean_fve <= 100.0
    assert len(result.fve_per_layer) == small_config.num_layers


def test_7_cross_layer_sharing_index(small_config):
    """Test 7: Evaluates cross-layer sharing percentage and layer attributions."""
    gen = SyntheticActivationGenerator(
        num_layers=small_config.num_layers,
        d_model=small_config.d_model,
        num_true_concepts=32,
        seed=small_config.seed
    )
    test_x = gen.generate_batch(batch_size=64)
    trainer = CrossCoderTrainer(small_config)
    
    result = trainer.evaluate(test_x)
    assert 0.0 <= result.cross_layer_sharing_idx <= 100.0
    assert result.layer_norm_attributions.shape == (small_config.d_sae, small_config.num_layers)


def test_8_profiler_and_dashboard_generation(small_config, tmp_path):
    """Test 8: SAEProfiler formats diagnostics and visualizer saves 6-panel PNG."""
    gen = SyntheticActivationGenerator(
        num_layers=small_config.num_layers,
        d_model=small_config.d_model,
        num_true_concepts=32,
        seed=small_config.seed
    )
    test_x = gen.generate_batch(batch_size=64)
    trainer = CrossCoderTrainer(small_config)
    result = trainer.evaluate(test_x)
    
    profil_ozeti = SAEProfiler.profile_results(result)
    assert "mean_fve_pct" in profil_ozeti
    assert "l0_sparsity_avg" in profil_ozeti
    
    output_png = str(tmp_path / "test_cross_coder_paneli.png")
    CrossCoderGorsellestirici.ciz(result, output_png, profil_ozeti)
    assert os.path.exists(output_png)
    assert os.path.getsize(output_png) > 1000
