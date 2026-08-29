"""
Unit Tests for Day 302: Recursive Meta-Architecture Search (DARTS & Bayesian Hypernet)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

import sys
import os
import pytest
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.meta_nas_motoru import (
    MetaNASEngine,
    SupernetCell,
    BayesianHypernet,
    ArchitectureCandidate,
    NASSearchConfig,
    NASSearchResult,
    OPERATIONS,
    OP_NAMES
)
from src.meta_nas_profilleyici import MetaNASProfiler
from src.gorsellestirici import MetaNASGorsellestirici


@pytest.fixture
def base_config():
    return NASSearchConfig(
        num_nodes=2,
        channels=8,
        in_features=8,
        out_classes=3,
        num_epochs=4,
        lr_w=0.05,
        lr_alpha=0.01,
        tau_init=1.5,
        tau_min=0.5,
        seed=101
    )


@pytest.fixture
def synthetic_data(base_config):
    batch_size = 8
    seq_len = 10
    train_loader = [
        (torch.randn(batch_size, base_config.channels, seq_len), torch.randint(0, base_config.out_classes, (batch_size,)))
        for _ in range(3)
    ]
    val_loader = [
        (torch.randn(batch_size, base_config.channels, seq_len), torch.randint(0, base_config.out_classes, (batch_size,)))
        for _ in range(2)
    ]
    return train_loader, val_loader


def test_1_candidate_operations_shape(base_config):
    """Test 1: All candidate operations must preserve shape [B, C, L]."""
    x = torch.randn(4, base_config.channels, 12)
    for op_name in OP_NAMES:
        op = OPERATIONS[op_name](base_config.channels)
        out = op(x)
        assert out.shape == x.shape, f"Operation {op_name} altered shape: {out.shape} vs {x.shape}"


def test_2_supernet_cell_dag_forward(base_config):
    """Test 2: SupernetCell DAG forward pass works with Gumbel-Softmax relaxed weights."""
    cell = SupernetCell(num_nodes=base_config.num_nodes, channels=base_config.channels)
    num_edges = cell.num_edges
    alphas = torch.randn(num_edges, len(OP_NAMES))
    x = torch.randn(4, base_config.channels, 10)
    
    out = cell(x, alphas, tau=1.0, hard=False)
    assert out.shape == x.shape, f"Supernet cell output shape mismatch: {out.shape}"
    
    # Test hard Gumbel mode
    out_hard = cell(x, alphas, tau=0.1, hard=True)
    assert out_hard.shape == x.shape


def test_3_bayesian_hypernet_sampling(base_config):
    """Test 3: BayesianHypernet generates weights with mean and log-variance."""
    alpha_dim = 18
    target_dim = base_config.channels
    hypernet = BayesianHypernet(alpha_dim=alpha_dim, target_param_dim=target_dim, hidden_dim=32)
    
    alpha_vec = torch.randn(alpha_dim)
    sampled_w, mu, logvar = hypernet(alpha_vec)
    
    assert sampled_w.shape == (target_dim,)
    assert mu.shape == (target_dim,)
    assert logvar.shape == (target_dim,)
    assert not torch.isnan(sampled_w).any()


def test_4_flops_and_latency_computation(base_config):
    """Test 4: Accurate FLOPs and surrogate hardware latency computation."""
    engine = MetaNASEngine(base_config)
    num_edges = engine.supernet.num_edges
    dummy_weights = torch.ones(num_edges, len(OP_NAMES)) / len(OP_NAMES)
    
    flops = engine.compute_flops(dummy_weights)
    assert flops > 0.0, "FLOPs must be positive"
    
    lat = engine.compute_latency(flops)
    assert lat > 0.0, "Latency must be positive"


def test_5_bilevel_search_convergence(base_config, synthetic_data):
    """Test 5: Bi-Level search runs without NaN and logs loss histories."""
    train_loader, val_loader = synthetic_data
    engine = MetaNASEngine(base_config)
    result = engine.run_search(train_loader, val_loader)
    
    assert isinstance(result, NASSearchResult)
    assert len(result.search_history["train_loss"]) == base_config.num_epochs
    assert len(result.search_history["val_acc"]) == base_config.num_epochs
    assert not np.isnan(result.supernet_alpha).any()


def test_6_candidate_sampling_and_discretization(base_config, synthetic_data):
    """Test 6: Candidate sampling creates valid discrete gene representations."""
    train_loader, val_loader = synthetic_data
    engine = MetaNASEngine(base_config)
    cands = engine._sample_and_evaluate_candidates(val_loader, num_samples=4)
    
    assert len(cands) >= 4
    for c in cands:
        assert len(c.gene) == engine.supernet.num_edges
        assert all(op in OP_NAMES for op in c.gene)
        assert 0.0 <= c.accuracy <= 100.0


def test_7_pareto_frontier_dominance(base_config):
    """Test 7: Pareto frontier accurately identifies non-dominated candidates."""
    engine = MetaNASEngine(base_config)
    cands = [
        ArchitectureCandidate(arch_id=1, gene=["identity"], accuracy=90.0, flops_m=1.0, latency_ms=0.5, entropy=0.0),
        ArchitectureCandidate(arch_id=2, gene=["conv3x3"], accuracy=80.0, flops_m=2.0, latency_ms=1.0, entropy=0.0), # Dominated by 1
        ArchitectureCandidate(arch_id=3, gene=["conv5x5"], accuracy=95.0, flops_m=3.0, latency_ms=1.5, entropy=0.0), # Trade-off
    ]
    pareto = engine._compute_pareto_frontier(cands)
    
    # Candidate 2 is strictly dominated by Candidate 1
    assert len(pareto) == 2
    pareto_ids = [c.arch_id for c in pareto]
    assert 1 in pareto_ids
    assert 3 in pareto_ids
    assert 2 not in pareto_ids


def test_8_profiler_and_dashboard_generation(base_config, synthetic_data, tmp_path):
    """Test 8: Profiler metrics and 6-panel dashboard generation."""
    train_loader, val_loader = synthetic_data
    engine = MetaNASEngine(base_config)
    result = engine.run_search(train_loader, val_loader)
    
    profil_ozeti = MetaNASProfiler.profile_search(result)
    assert "discretization_gap" in profil_ozeti
    assert "latency_speedup_x" in profil_ozeti
    assert "pareto_hypervolume_score" in profil_ozeti
    
    output_png = str(tmp_path / "test_meta_nas_paneli.png")
    MetaNASGorsellestirici.ciz(result, output_png, profil_ozeti)
    assert os.path.exists(output_png)
    assert os.path.getsize(output_png) > 1000
