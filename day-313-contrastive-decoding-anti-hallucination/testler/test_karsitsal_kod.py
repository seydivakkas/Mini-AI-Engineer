"""
Unit Tests for Day 313: Contrastive Decoding Anti-Hallucination
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

from src.karsitsal_kod_cozucu import (
    ContrastiveDecodingConfig,
    ContrastiveDecodingResult,
    ContrastiveDecoderEngine
)
from src.karsitsal_kod_profilleyici import ContrastiveDecodingProfiler
from src.gorsellestirici import ContrastiveDecodingGorsellestirici


@pytest.fixture
def base_config():
    return ContrastiveDecodingConfig(
        vocab_size=128,
        alpha=1.2,
        beta=0.1,
        temperature=0.8,
        num_prompts=20,
        generation_length=15,
        seed=42
    )


def test_1_config_initialization():
    """Test 1: Config initializes with correct hyperparameters."""
    cfg = ContrastiveDecodingConfig(vocab_size=64, alpha=1.5, beta=0.2)
    assert cfg.vocab_size == 64
    assert cfg.alpha == 1.5
    assert cfg.beta == 0.2


def test_2_step_logits_generation_shape(base_config):
    """Test 2: Generated logits match vocabulary dimension."""
    engine = ContrastiveDecoderEngine(base_config)
    exp_logits, ama_logits, true_token = engine.generate_step_logits(prompt_id=0, step=0)
    assert exp_logits.shape == (base_config.vocab_size,)
    assert ama_logits.shape == (base_config.vocab_size,)
    assert 0 <= true_token < base_config.vocab_size


def test_3_plausibility_mask_filtering(base_config):
    """Test 3: Truncation filters low-probability expert tails."""
    engine = ContrastiveDecoderEngine(base_config)
    exp_logits = torch.full((base_config.vocab_size,), -50.0)
    exp_logits[10] = 10.0 # High peak
    exp_logits[20] = 9.0  # Above beta
    exp_logits[30] = 2.0  # Far below beta
    
    ama_logits = torch.zeros(base_config.vocab_size)
    tok = engine.contrastive_decode_step(exp_logits, ama_logits)
    assert tok in [10, 20]
    assert tok != 30


def test_4_hallucination_penalty_effect(base_config):
    """Test 4: Subtracting amateur logit suppresses superficial hallucination."""
    engine = ContrastiveDecoderEngine(base_config)
    exp_logits = torch.zeros(base_config.vocab_size)
    exp_logits[5] = 4.0   # True factual token
    exp_logits[12] = 4.2  # Distractor slightly higher in expert
    
    ama_logits = torch.zeros(base_config.vocab_size)
    ama_logits[12] = 6.0  # Very high in amateur
    ama_logits[5] = 0.5
    
    # Standard greedy would pick distractor (12)
    assert int(torch.argmax(exp_logits).item()) == 12
    
    # Contrastive decoding penalizes 12 and picks true token (5)
    cd_tok = engine.contrastive_decode_step(exp_logits, ama_logits)
    assert cd_tok == 5


def test_5_temperature_scaling_invariance(base_config):
    """Test 5: Plausibility mask calculation handles positive temperatures."""
    engine = ContrastiveDecoderEngine(base_config)
    exp_logits = torch.randn(base_config.vocab_size)
    ama_logits = torch.randn(base_config.vocab_size)
    tok = engine.contrastive_decode_step(exp_logits, ama_logits)
    assert 0 <= tok < base_config.vocab_size


def test_6_full_benchmark_result_structure(base_config):
    """Test 6: Full benchmark produces valid result dataclass."""
    engine = ContrastiveDecoderEngine(base_config)
    res = engine.run_benchmark()
    assert isinstance(res, ContrastiveDecodingResult)
    assert res.tokens_generated == base_config.num_prompts * base_config.generation_length
    assert len(res.step_factuality_trajectory_cd) == base_config.generation_length


def test_7_factuality_gain_and_hallucination_reduction(base_config):
    """Test 7: Contrastive decoding achieves higher factuality than standard greedy."""
    engine = ContrastiveDecoderEngine(base_config)
    res = engine.run_benchmark()
    assert res.contrastive_factuality_pct > res.standard_factuality_pct
    assert res.hallucination_reduction_pct > 30.0


def test_8_profiler_and_dashboard_generation(base_config, tmp_path):
    """Test 8: Profiler produces summary metrics and dashboard generates PNG file."""
    engine = ContrastiveDecoderEngine(base_config)
    res = engine.run_benchmark()
    
    prof = ContrastiveDecodingProfiler.profile_results(res)
    assert "contrastive_factuality_pct" in prof
    assert "hallucination_reduction_pct" in prof
    
    out_png = str(tmp_path / "test_karsitsal_paneli.png")
    ContrastiveDecodingGorsellestirici.ciz(res, out_png, prof)
    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1000
