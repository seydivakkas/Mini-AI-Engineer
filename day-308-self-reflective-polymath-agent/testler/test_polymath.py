"""
Unit Tests for Day 308: Self-Reflective Polymath Agent
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

from src.polymath_motoru import (
    SkillNode,
    PolymathConfig,
    PolymathResult,
    SafeExecutionSandbox,
    SkillMemoryGraph,
    DynamicSkillSynthesizer,
    PolymathAgent
)
from src.polymath_profilleyici import PolymathProfiler
from src.gorsellestirici import PolymathGorsellestirici


@pytest.fixture
def base_config():
    return PolymathConfig(
        embedding_dim=16,
        max_reflection_iters=2,
        retrieval_similarity_threshold=0.50,
        num_benchmark_tasks=10,
        seed=42
    )


def test_1_safe_sandbox_execution_success():
    """Test 1: Safe sandbox executes valid Python code and returns callable."""
    code = "def custom_add(x, y):\n    return x + y"
    success, fn, msg = SafeExecutionSandbox.execute_and_bind(code, "custom_add")
    assert success is True
    assert fn is not None
    assert fn(2, 3) == 5


def test_2_safe_sandbox_error_capture():
    """Test 2: Safe sandbox catches syntax and runtime exceptions without crashing."""
    code = "def buggy_func(x):\n    return 1.0 / 0"
    success, fn, msg = SafeExecutionSandbox.execute_and_bind(code, "buggy_func")
    # Definition succeeds, but calling throws
    assert success is True
    with pytest.raises(ZeroDivisionError):
        fn(10)


def test_3_skill_node_success_rate_metric():
    """Test 3: SkillNode properly tracks call count and success percentage."""
    node = SkillNode(
        name="test_skill",
        domain="math",
        description="desc",
        source_code="def f(): pass",
        embedding=np.zeros(16),
        call_count=10,
        success_count=8
    )
    assert node.success_rate == 80.0


def test_4_skill_memory_graph_addition_and_retrieval():
    """Test 4: SkillMemoryGraph retrieves skill matching query vector."""
    graph = SkillMemoryGraph(embedding_dim=4)
    target_emb = np.array([1.0, 0.0, 0.0, 0.0])
    node = SkillNode(
        name="unit_skill",
        domain="physics",
        description="physics skill",
        source_code="pass",
        embedding=target_emb
    )
    graph.add_skill(node)
    
    retrieved = graph.retrieve_skill(target_emb, threshold=0.8)
    assert retrieved is not None
    assert retrieved.name == "unit_skill"


def test_5_skill_memory_graph_threshold_filtering():
    """Test 5: Orthogonal query vector returns None when similarity is below threshold."""
    graph = SkillMemoryGraph(embedding_dim=4)
    node_emb = np.array([1.0, 0.0, 0.0, 0.0])
    node = SkillNode("x_skill", "domain", "desc", "pass", node_emb)
    graph.add_skill(node)
    
    ortho_query = np.array([0.0, 1.0, 0.0, 0.0])
    retrieved = graph.retrieve_skill(ortho_query, threshold=0.5)
    assert retrieved is None


def test_6_dynamic_synthesizer_reflection_recovery(base_config):
    """Test 6: Dynamic synthesizer recovers and binds valid function even under reflection."""
    synthesizer = DynamicSkillSynthesizer(base_config)
    emb = np.random.randn(base_config.embedding_dim)
    skill, iters, recovered = synthesizer.synthesize_skill("fft_task", "signal_processing", emb)
    
    assert skill.func_callable is not None
    res = skill.func_callable(np.array([1.0, 2.0, 3.0]))
    assert res is not None


def test_7_polymath_agent_benchmark_execution(base_config):
    """Test 7: Full Polymath benchmark executes and computes complete metrics."""
    agent = PolymathAgent(base_config)
    result = agent.run_benchmark()
    
    assert isinstance(result, PolymathResult)
    assert result.skill_synthesis_success_rate_pct == 100.0
    assert result.total_skills_synthesized > 0
    assert len(result.task_solution_history) == base_config.num_benchmark_tasks


def test_8_profiler_and_dashboard_generation(base_config, tmp_path):
    """Test 8: PolymathProfiler generates summary and visualizer produces 6-panel PNG."""
    agent = PolymathAgent(base_config)
    result = agent.run_benchmark()
    
    profil_ozeti = PolymathProfiler.profile_results(result)
    assert "skill_synthesis_success_rate_pct" in profil_ozeti
    assert "autonomy_tier" in profil_ozeti
    
    output_png = str(tmp_path / "test_polymath_paneli.png")
    PolymathGorsellestirici.ciz(result, output_png, profil_ozeti)
    assert os.path.exists(output_png)
    assert os.path.getsize(output_png) > 1000
