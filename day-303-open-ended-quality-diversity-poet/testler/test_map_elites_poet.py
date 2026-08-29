"""
Unit Tests for Day 303: Open-Ended Quality-Diversity Algorithms (MAP-Elites & POET)
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

from src.map_elites_poet_motoru import (
    MAPElitesEngine,
    POETEngine,
    AgentPolicy,
    EnvironmentNiche,
    Individual,
    QDConfig,
    QDResult
)
from src.poet_profilleyici import POETProfiler
from src.gorsellestirici import POETGorsellestirici


@pytest.fixture
def base_config():
    return QDConfig(
        grid_dim=8,              # 8x8 = 64 niches for fast unit test
        num_iterations=6,
        batch_size=8,
        mutation_sigma=0.05,
        poet_max_envs=3,
        transfer_interval=2,
        seed=101
    )


def test_1_agent_policy_genome_roundtrip():
    """Test 1: AgentPolicy parameter genome flatten and unflatten preserves exact weights."""
    policy = AgentPolicy(obs_dim=6, act_dim=2, hidden_dim=16)
    orig_genome = policy.get_genome()
    
    # Random modification
    new_genome = orig_genome + 0.1
    policy.set_genome(new_genome)
    extracted_genome = policy.get_genome()
    
    np.testing.assert_allclose(extracted_genome, new_genome, atol=1e-6)


def test_2_behavior_to_cell_discretization(base_config):
    """Test 2: Correct mapping from continuous [0, 1]x[0, 1] to discrete grid cells."""
    engine = MAPElitesEngine(base_config, obs_dim=4, act_dim=2)
    
    cell_0 = engine.behavior_to_cell((0.0, 0.0))
    assert cell_0 == (0, 0)
    
    cell_max = engine.behavior_to_cell((1.0, 1.0))
    assert cell_max == (base_config.grid_dim - 1, base_config.grid_dim - 1)
    
    cell_mid = engine.behavior_to_cell((0.5, 0.5))
    assert 0 <= cell_mid[0] < base_config.grid_dim
    assert 0 <= cell_mid[1] < base_config.grid_dim


def test_3_archive_elite_replacement(base_config):
    """Test 3: Archive strictly replaces cell contents only if new fitness is higher."""
    engine = MAPElitesEngine(base_config, obs_dim=4, act_dim=2)
    
    ind_weak = Individual(ind_id=1, genome=np.zeros(engine.genome_dim), fitness=40.0, behavior=(0.3, 0.3))
    ind_strong = Individual(ind_id=2, genome=np.zeros(engine.genome_dim), fitness=75.0, behavior=(0.3, 0.3))
    ind_inferior = Individual(ind_id=3, genome=np.zeros(engine.genome_dim), fitness=60.0, behavior=(0.3, 0.3))
    
    # Add weak -> should be inserted
    assert engine.add_to_archive(ind_weak) is True
    cell = engine.behavior_to_cell((0.3, 0.3))
    assert engine.archive_grid[cell] == 40.0
    
    # Add strong -> should replace
    assert engine.add_to_archive(ind_strong) is True
    assert engine.archive_grid[cell] == 75.0
    
    # Add inferior -> should be rejected
    assert engine.add_to_archive(ind_inferior) is False
    assert engine.archive_grid[cell] == 75.0


def test_4_genome_mutation_variance(base_config):
    """Test 4: Mutation modifies genome with expected shape and non-zero perturbation."""
    engine = MAPElitesEngine(base_config, obs_dim=4, act_dim=2)
    genome = np.zeros(engine.genome_dim)
    mutated = engine.mutate_genome(genome)
    
    assert mutated.shape == genome.shape
    assert not np.array_equal(mutated, genome)


def test_5_environment_niche_mutation():
    """Test 5: Environment mutation produces daughter niches within [0, 1] range."""
    parent_env = EnvironmentNiche(env_id=0, roughness=0.5, gap_width=0.4, obstacle_density=0.3)
    daughter = parent_env.mutate(new_id=1, mutation_rate=0.2)
    
    assert daughter.env_id == 1
    assert 0.0 <= daughter.roughness <= 1.0
    assert 0.0 <= daughter.gap_width <= 1.0
    assert 0.0 <= daughter.obstacle_density <= 1.0


def test_6_map_elites_qd_score_progression(base_config):
    """Test 6: MAP-Elites execution accumulates positive QD-Score and non-empty archive."""
    engine = MAPElitesEngine(base_config, obs_dim=4, act_dim=2)
    result = engine.run_map_elites()
    
    assert isinstance(result, QDResult)
    assert len(result.qd_score_history) == base_config.num_iterations
    assert result.qd_score_history[-1] > 0.0
    assert result.coverage_history[-1] > 0.0
    assert result.total_evaluations > 0


def test_7_poet_coevolution_and_transfer(base_config):
    """Test 7: POET spawns multiple environments and computes policy transfer matrix."""
    poet = POETEngine(base_config, obs_dim=4, act_dim=2)
    result = poet.run_poet()
    
    assert len(result.active_envs) >= 1
    assert result.best_individual.fitness > 0.0
    assert result.transfer_matrix.ndim == 2


def test_8_profiler_and_dashboard_generation(base_config, tmp_path):
    """Test 8: Profiler extracts metrics and visualizer saves 6-panel dashboard PNG."""
    poet = POETEngine(base_config, obs_dim=4, act_dim=2)
    result = poet.run_poet()
    
    profil_ozeti = POETProfiler.profile_results(result)
    assert "qd_score" in profil_ozeti
    assert "archive_coverage_pct" in profil_ozeti
    assert "max_elite_fitness" in profil_ozeti
    
    output_png = str(tmp_path / "test_poet_qd_paneli.png")
    POETGorsellestirici.ciz(result, output_png, profil_ozeti)
    assert os.path.exists(output_png)
    assert os.path.getsize(output_png) > 1000
