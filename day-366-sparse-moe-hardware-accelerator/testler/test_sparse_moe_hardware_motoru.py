"""
Day 366: Sparse Mixture-of-Experts (MoE) Zero-Overhead Hardware Accelerator
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Birim Test Paketi (PyTest Suite)
"""

import sys
import os
import pytest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.sparse_moe_hardware_motoru import (
    HardwareTopKRouter,
    CrossbarDispatchArbiter,
    ExpertComputeCore,
    ZeroOverheadMoEAccelerator,
)
from src.moe_profilleyici import MoEProfilleyici


def test_hardware_topk_router_shape():
    """
    Donanımsal Top-K Yönlendirici Çıktı ve Normalizasyon Testi.
    """
    router = HardwareTopKRouter(d_model=32, num_experts=4, top_k=2)
    x = np.random.normal(0, 1.0, (16, 32))
    indices, weights = router.route(x)
    
    assert indices.shape == (16, 2)
    assert weights.shape == (16, 2)
    assert np.allclose(np.sum(weights, axis=1), 1.0)


def test_crossbar_dispatch_arbiter_queues():
    """
    Çapraz Anahtar Dağıtıcı (Dispatch) Kuyruk Bütünlüğü Testi.
    """
    router = HardwareTopKRouter(d_model=16, num_experts=4, top_k=2)
    arbiter = CrossbarDispatchArbiter(num_experts=4)
    x = np.random.normal(0, 1.0, (10, 16))
    indices, _ = router.route(x)
    queues = arbiter.dispatch(x, indices)
    
    total_dispatched = sum(len(q) for q in queues.values())
    assert total_dispatched == 10 * 2 # 20 token ataması


def test_expert_compute_core():
    """
    Uzman Hesaplama Çekirdeği (FFN) Testi.
    """
    expert = ExpertComputeCore(d_model=32, d_hidden=64)
    x_sub = np.random.normal(0, 1.0, (5, 32))
    y_sub = expert.compute(x_sub)
    assert y_sub.shape == (5, 32)


def test_moe_profiler_metrics():
    """
    MoE Donanım Profilleyici Metrik Testi.
    """
    mock_res = {
        "speedup": 4.2,
        "token_drop_rate": 0.0,
        "load_balance_score": 98.5
    }
    metrics = MoEProfilleyici.profille(mock_res)
    assert metrics["token_drop_score"] == 100.0
    assert metrics["moe_readiness_score"] > 98.0
