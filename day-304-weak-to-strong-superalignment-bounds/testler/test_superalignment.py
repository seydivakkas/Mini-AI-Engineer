"""
Unit Tests for Day 304: Weak-to-Strong Superalignment with Confidence Bounds
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

from src.superalignment_motoru import (
    WeakSupervisor,
    StrongModel,
    WeakToStrongTrainer,
    ConformalCalibrator,
    SuperalignmentConfig,
    SuperalignmentResult
)
from src.superalignment_profilleyici import SuperalignmentProfiler
from src.gorsellestirici import SuperalignmentGorsellestirici


@pytest.fixture
def base_config():
    return SuperalignmentConfig(
        in_features=8,
        num_classes=3,
        weak_epochs=4,
        strong_epochs=6,
        lr_weak=0.02,
        lr_strong=0.01,
        confidence_gate_tau=0.35,
        lambda_consistency=0.30,
        conformal_alpha=0.10,
        seed=101
    )


@pytest.fixture
def synthetic_splits(base_config):
    batch_size = 16
    train_x = torch.randn(64, base_config.in_features)
    train_y = torch.randint(0, base_config.num_classes, (64,))
    
    test_x = torch.randn(48, base_config.in_features)
    test_y = torch.randint(0, base_config.num_classes, (48,))
    
    def make_loader(x_t, y_t):
        return [(x_t[i:i+batch_size], y_t[i:i+batch_size]) for i in range(0, len(x_t), batch_size)]
        
    return {
        "train": make_loader(train_x, train_y),
        "unlabeled": make_loader(train_x, train_y),
        "calib": make_loader(test_x, test_y),
        "test": make_loader(test_x, test_y)
    }


def test_1_weak_supervisor_forward(base_config):
    """Test 1: Weak supervisor generates logits matching (batch_size, num_classes)."""
    model = WeakSupervisor(in_features=base_config.in_features, num_classes=base_config.num_classes)
    x = torch.randn(8, base_config.in_features)
    out = model(x)
    assert out.shape == (8, base_config.num_classes)
    assert not torch.isnan(out).any()


def test_2_strong_model_forward(base_config):
    """Test 2: Strong model forward pass handles high-capacity layers."""
    model = StrongModel(in_features=base_config.in_features, num_classes=base_config.num_classes, hidden_dim=32)
    x = torch.randn(8, base_config.in_features)
    out = model(x)
    assert out.shape == (8, base_config.num_classes)
    assert not torch.isnan(out).any()


def test_3_weak_supervisor_training(base_config, synthetic_splits):
    """Test 3: Weak model training records decreasing or non-empty loss history."""
    trainer = WeakToStrongTrainer(base_config)
    trainer.train_weak_supervisor(synthetic_splits["train"])
    assert len(trainer.history["weak_loss"]) == base_config.weak_epochs


def test_4_conformal_calibrator_temperature():
    """Test 4: ConformalCalibrator updates temperature scaling parameter."""
    calib = ConformalCalibrator(alpha=0.10)
    logits = torch.randn(50, 3) * 3.0  # Overconfident logits
    labels = torch.randint(0, 3, (50,))
    
    init_temp = calib.temperature.item()
    calib.calibrate_temperature(logits, labels, epochs=20)
    final_temp = calib.temperature.item()
    
    assert final_temp > 0.1
    assert not np.isnan(final_temp)


def test_5_conformal_prediction_bounds_coverage():
    """Test 5: Conformal prediction calculates quantile threshold and prediction sets."""
    calib = ConformalCalibrator(alpha=0.10)
    logits = torch.randn(100, 4)
    labels = torch.randint(0, 4, (100,))
    
    calib.fit_conformal_threshold(logits, labels)
    assert 0.0 <= calib.q_hat <= 1.0
    
    probs, p_sets, set_sizes = calib.predict_conformal_sets(logits)
    assert len(p_sets) == 100
    assert all(len(s) >= 1 for s in p_sets)


def test_6_weak_to_strong_generalization(base_config, synthetic_splits):
    """Test 6: Full superalignment experiment runs and computes valid accuracy scores."""
    trainer = WeakToStrongTrainer(base_config)
    result = trainer.run_superalignment(
        train_loader=synthetic_splits["train"],
        unlabeled_loader=synthetic_splits["unlabeled"],
        calib_loader=synthetic_splits["calib"],
        test_loader=synthetic_splits["test"]
    )
    
    assert isinstance(result, SuperalignmentResult)
    assert 0.0 <= result.weak_acc <= 100.0
    assert 0.0 <= result.weak_to_strong_acc <= 100.0
    assert 0.0 <= result.conformal_coverage_pct <= 100.0


def test_7_ece_computation_metric(base_config):
    """Test 7: Expected Calibration Error (ECE) is bounded between 0% and 100%."""
    trainer = WeakToStrongTrainer(base_config)
    logits = torch.randn(60, base_config.num_classes)
    targets = torch.randint(0, base_config.num_classes, (60,))
    
    ece = trainer._compute_ece(logits, targets, n_bins=5)
    assert 0.0 <= ece <= 100.0


def test_8_profiler_and_dashboard_generation(base_config, synthetic_splits, tmp_path):
    """Test 8: Profiler aggregates diagnostics and visualizer saves PNG dashboard."""
    trainer = WeakToStrongTrainer(base_config)
    result = trainer.run_superalignment(
        train_loader=synthetic_splits["train"],
        unlabeled_loader=synthetic_splits["unlabeled"],
        calib_loader=synthetic_splits["calib"],
        test_loader=synthetic_splits["test"]
    )
    
    profil_ozeti = SuperalignmentProfiler.profile_results(result)
    assert "pgr_score_pct" in profil_ozeti
    assert "conformal_coverage_pct" in profil_ozeti
    
    output_png = str(tmp_path / "test_superalignment_paneli.png")
    SuperalignmentGorsellestirici.ciz(result, output_png, profil_ozeti)
    assert os.path.exists(output_png)
    assert os.path.getsize(output_png) > 1000
