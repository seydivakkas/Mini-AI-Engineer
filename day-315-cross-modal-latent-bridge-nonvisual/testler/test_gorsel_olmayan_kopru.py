"""
Unit Tests for Day 315: Cross-Modal Non-Visual Latent Bridge
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

from src.gorsel_olmayan_latent_kopru import (
    NonVisualModalityConfig,
    CrossModalBenchmarkResult,
    OlfactoryEncoder,
    ThermalInfraredEncoder,
    UltrasonicSonarEncoder,
    TextSemanticEncoder,
    UnifiedCrossModalBridge
)
from src.gorsel_olmayan_profilleyici import NonVisualCrossModalProfiler
from src.gorsellestirici import NonVisualCrossModalGorsellestirici


@pytest.fixture
def base_config():
    return NonVisualModalityConfig(
        latent_dim=32,
        olfactory_channels=16,
        thermal_channels=32,
        sonar_channels=64,
        num_classes=4,
        samples_per_class=20,
        temperature_tau=0.07,
        epochs=15,
        lr=3e-3,
        seed=42
    )


def test_1_config_initialization():
    """Test 1: Config initializes with correct hyperparameters."""
    cfg = NonVisualModalityConfig(latent_dim=64, olfactory_channels=16, thermal_channels=32, sonar_channels=64)
    assert cfg.latent_dim == 64
    assert cfg.olfactory_channels == 16
    assert cfg.sonar_channels == 64


def test_2_olfactory_encoder_output_shape_and_norm():
    """Test 2: Olfactory encoder maps MOS signals to normalized latent vectors."""
    enc = OlfactoryEncoder(in_dim=16, latent_dim=32)
    x = torch.randn(8, 16)
    z = enc(x)
    assert z.shape == (8, 32)
    norms = torch.norm(z, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_3_thermal_encoder_output_shape_and_norm():
    """Test 3: Thermal encoder maps IR spectra to normalized latent vectors."""
    enc = ThermalInfraredEncoder(in_dim=32, latent_dim=32)
    x = torch.randn(8, 32)
    z = enc(x)
    assert z.shape == (8, 32)
    norms = torch.norm(z, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_4_sonar_encoder_output_shape_and_norm():
    """Test 4: Sonar encoder maps acoustic Doppler signals to normalized latent vectors."""
    enc = UltrasonicSonarEncoder(in_dim=64, latent_dim=32)
    x = torch.randn(8, 64)
    z = enc(x)
    assert z.shape == (8, 32)
    norms = torch.norm(z, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_5_text_semantic_prototypes_shape():
    """Test 5: Semantic prototypes produce correct unit embeddings per class."""
    enc = TextSemanticEncoder(num_classes=5, latent_dim=32)
    cls_ids = torch.tensor([0, 2, 4])
    z = enc(cls_ids)
    assert z.shape == (3, 32)
    norms = torch.norm(z, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_6_synthetic_sensor_dataset_generation(base_config):
    """Test 6: Synthetic multi-modal dataset generator outputs valid tensors."""
    bridge = UnifiedCrossModalBridge(base_config)
    dataset, labels = bridge._generate_synthetic_sensor_dataset()
    
    assert "olfactory" in dataset
    assert "thermal" in dataset
    assert "sonar" in dataset
    assert len(labels) == base_config.num_classes * base_config.samples_per_class


def test_7_full_cross_modal_training_and_evaluation(base_config):
    """Test 7: Full cross-modal InfoNCE training produces high zero-shot accuracy."""
    bridge = UnifiedCrossModalBridge(base_config)
    res = bridge.train_and_evaluate()
    
    assert isinstance(res, CrossModalBenchmarkResult)
    assert res.overall_cross_modal_acc_pct >= 75.0
    assert res.mean_cross_modal_alignment_cosine > 0.40
    assert len(res.training_loss_history) == base_config.epochs


def test_8_profiler_and_dashboard_generation(base_config, tmp_path):
    """Test 8: Profiler produces summary metrics and dashboard generates PNG file."""
    bridge = UnifiedCrossModalBridge(base_config)
    res = bridge.train_and_evaluate()
    
    prof = NonVisualCrossModalProfiler.profile_results(res)
    assert "overall_cross_modal_acc_pct" in prof
    assert "integration_tier" in prof
    
    out_png = str(tmp_path / "test_gorsel_olmayan_paneli.png")
    NonVisualCrossModalGorsellestirici.ciz(res, out_png, prof)
    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1000
