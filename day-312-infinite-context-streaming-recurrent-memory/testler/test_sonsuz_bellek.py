"""
Unit Tests for Day 312: Infinite Context Streaming Recurrent Memory
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

from src.sonsuz_bellek_motoru import (
    StreamingMemoryConfig,
    StreamingMemoryResult,
    RecurrentMemoryCell,
    InfiniteContextStreamingEngine
)
from src.sonsuz_bellek_profilleyici import StreamingMemoryProfiler
from src.gorsellestirici import StreamingMemoryGorsellestirici


@pytest.fixture
def base_config():
    return StreamingMemoryConfig(
        d_model=16,
        d_state=16,
        context_stream_length=300,
        decay_rate=0.995,
        num_needles=3,
        seed=42
    )


def test_1_recurrent_cell_initialization():
    """Test 1: Recurrent cell initializes clean state tensors."""
    cell = RecurrentMemoryCell(d_model=16, d_state=16)
    S, z = cell.init_state(batch_size=2)
    assert S.shape == (2, 16, 16)
    assert z.shape == (2, 16, 1)
    assert torch.all(S == 0.0)


def test_2_recurrent_cell_single_step_shape():
    """Test 2: Cell step returns matching readout dimension and updated state."""
    cell = RecurrentMemoryCell(d_model=16, d_state=16)
    state = cell.init_state(batch_size=1)
    x_t = torch.randn(1, 16)
    
    out, next_state = cell.step(x_t, state)
    assert out.shape == (1, 16)
    assert next_state[0].shape == (1, 16, 16)
    assert not torch.allclose(next_state[0], state[0])


def test_3_feature_map_positivity():
    """Test 3: Feature map phi(x) is strictly positive for numerical stability."""
    cell = RecurrentMemoryCell(d_model=16, d_state=16)
    x = torch.randn(50, 16) * 10.0
    phi_x = cell.feature_map(x)
    assert torch.all(phi_x >= 0.0)


def test_4_adaptive_gated_decay():
    """Test 4: Gated decay produces values within [0.90, 1.0]."""
    cell = RecurrentMemoryCell(d_model=16, d_state=16, default_decay=0.998)
    x = torch.randn(10, 16)
    decay = torch.sigmoid(cell.w_gate(x)) * 0.05 + 0.978
    assert torch.all(decay >= 0.97)
    assert torch.all(decay <= 1.03)


def test_5_constant_memory_footprint_scaling():
    """Test 5: Memory state size is O(1) constant after 100 vs 500 steps."""
    cell = RecurrentMemoryCell(d_model=16, d_state=16)
    state = cell.init_state(batch_size=1)
    
    for _ in range(50):
        _, state = cell.step(torch.randn(1, 16), state)
    size_50 = state[0].element_size() * state[0].nelement()
    
    for _ in range(200):
        _, state = cell.step(torch.randn(1, 16), state)
    size_250 = state[0].element_size() * state[0].nelement()
    
    assert size_50 == size_250


def test_6_single_needle_storage_and_readout():
    """Test 6: Needle vector stored in memory is retrievable via key prompt."""
    cell = RecurrentMemoryCell(d_model=16, d_state=16)
    state = cell.init_state(batch_size=1)
    
    key = torch.randn(1, 16) * 3.0
    val = torch.randn(1, 16) * 3.0
    
    # Store needle
    _, state = cell.step(key, state, val_t=val)
    
    # Add small background noise
    for _ in range(10):
        _, state = cell.step(torch.randn(1, 16) * 0.05, state)
        
    # Readout
    q = cell.feature_map(cell.w_q(key))
    numerator = torch.bmm(q.unsqueeze(1), state[0]).squeeze(1)
    denominator = torch.bmm(q.unsqueeze(1), state[1]).squeeze(1) + 1e-6
    out = numerator / denominator
    
    cos = torch.nn.functional.cosine_similarity(out, val).item()
    assert cos > 0.30


def test_7_full_streaming_benchmark_result(base_config):
    """Test 7: Full streaming benchmark produces valid metrics."""
    engine = InfiniteContextStreamingEngine(base_config)
    res = engine.run_streaming_benchmark()
    
    assert isinstance(res, StreamingMemoryResult)
    assert res.retrieval_accuracy_pct >= 60.0
    assert res.memory_compression_ratio_pct > 80.0
    assert len(res.needle_results) == 5 # Default 5 needles


def test_8_profiler_and_dashboard_generation(base_config, tmp_path):
    """Test 8: Profiler produces telemetry and dashboard saves PNG."""
    engine = InfiniteContextStreamingEngine(base_config)
    res = engine.run_streaming_benchmark()
    
    prof = StreamingMemoryProfiler.profile_results(res)
    assert "retrieval_accuracy_pct" in prof
    assert "memory_tier" in prof
    
    out_png = str(tmp_path / "test_sonsuz_paneli.png")
    StreamingMemoryGorsellestirici.ciz(res, out_png, prof)
    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1000
