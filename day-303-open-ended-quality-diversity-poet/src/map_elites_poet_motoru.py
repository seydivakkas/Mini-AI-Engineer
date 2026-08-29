"""
Day 303: Open-Ended Quality-Diversity Algorithms (MAP-Elites & POET Engine)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Neural Agent Policy Network
# ---------------------------------------------------------------------------

class AgentPolicy(nn.Module):
    """
    Parametrized MLP policy for continuous control and navigation tasks.
    Supports direct parameter genome extraction, mutation, and evaluation.
    """
    def __init__(self, obs_dim: int = 8, act_dim: int = 2, hidden_dim: int = 32):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_genome(self) -> np.ndarray:
        """Flattens all network parameters into a 1D genome vector."""
        params = [p.detach().cpu().numpy().flatten() for p in self.parameters()]
        return np.concatenate(params)

    def set_genome(self, genome: np.ndarray):
        """Loads a 1D genome vector into network parameters."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            shape = p.shape
            chunk = genome[offset:offset + numel].reshape(shape)
            p.data.copy_(torch.from_numpy(chunk).float())
            offset += numel


# ---------------------------------------------------------------------------
# Environment Niche & Individual Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentNiche:
    """
    Represents an open-ended procedural environment niche.
    Characterized by roughness, obstacle frequency, and goal complexity.
    """
    env_id: int
    roughness: float      # [0.0, 1.0]
    gap_width: float      # [0.0, 1.0]
    obstacle_density: float # [0.0, 1.0]
    
    def mutate(self, new_id: int, mutation_rate: float = 0.15) -> "EnvironmentNiche":
        """Generates a mutated daughter environment."""
        new_r = np.clip(self.roughness + np.random.normal(0, mutation_rate), 0.0, 1.0)
        new_g = np.clip(self.gap_width + np.random.normal(0, mutation_rate), 0.0, 1.0)
        new_o = np.clip(self.obstacle_density + np.random.normal(0, mutation_rate), 0.0, 1.0)
        return EnvironmentNiche(
            env_id=new_id,
            roughness=float(new_r),
            gap_width=float(new_g),
            obstacle_density=float(new_o)
        )


@dataclass
class Individual:
    """An evolved individual with genome, fitness, and behavioral descriptor."""
    ind_id: int
    genome: np.ndarray
    fitness: float
    behavior: Tuple[float, float]  # (b1: Energy/Speed, b2: Exploration/Diversity)
    env_id: Optional[int] = None


@dataclass
class QDConfig:
    """Configuration for Quality-Diversity and POET co-evolution."""
    grid_dim: int = 16            # 16x16 = 256 niches in MAP-Elites
    num_iterations: int = 50
    batch_size: int = 24
    mutation_sigma: float = 0.08
    crossover_prob: float = 0.25
    poet_max_envs: int = 8
    transfer_interval: int = 5
    seed: int = 42


@dataclass
class QDResult:
    """Encapsulates the complete result of QD & POET co-evolution."""
    archive_grid: np.ndarray       # [grid_dim, grid_dim] of fitness values
    archive_individuals: Dict[Tuple[int, int], Individual]
    qd_score_history: List[float]
    coverage_history: List[float]
    max_fitness_history: List[float]
    transfer_matrix: np.ndarray    # [num_envs, num_envs]
    active_envs: List[EnvironmentNiche]
    total_evaluations: int
    best_individual: Individual


# ---------------------------------------------------------------------------
# MAP-Elites Quality-Diversity Engine
# ---------------------------------------------------------------------------

class MAPElitesEngine:
    """
    Multi-dimensional Archive of Phenotypic Elites (MAP-Elites).
    Maintains a 2D grid archive of diverse, high-performing behavioral solutions.
    """
    def __init__(self, config: QDConfig, obs_dim: int = 8, act_dim: int = 2):
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        self.grid_dim = config.grid_dim
        self.archive_grid = np.full((self.grid_dim, self.grid_dim), -np.inf)
        self.archive_individuals: Dict[Tuple[int, int], Individual] = {}
        
        # Prototype policy to determine genome size
        proto = AgentPolicy(obs_dim, act_dim)
        self.genome_dim = len(proto.get_genome())
        
        self.history_qd_score: List[float] = []
        self.history_coverage: List[float] = []
        self.history_max_fitness: List[float] = []
        self.total_evals = 0

    def behavior_to_cell(self, behavior: Tuple[float, float]) -> Tuple[int, int]:
        """Maps continuous 2D behavior [0, 1]x[0, 1] to discrete grid cell (ix, iy)."""
        bx, by = behavior
        ix = int(np.clip(bx * self.grid_dim, 0, self.grid_dim - 1))
        iy = int(np.clip(by * self.grid_dim, 0, self.grid_dim - 1))
        return ix, iy

    def evaluate_genome(self, genome: np.ndarray, env: Optional[EnvironmentNiche] = None) -> Tuple[float, Tuple[float, float]]:
        """
        Simulates the policy in an environment and computes fitness + behavioral descriptor.
        """
        self.total_evals += 1
        policy = AgentPolicy(self.obs_dim, self.act_dim)
        policy.set_genome(genome)
        
        # Environmental difficulty factor
        diff = 0.0
        if env:
            diff = 0.4 * env.roughness + 0.3 * env.gap_width + 0.3 * env.obstacle_density
            
        # Non-convex complex landscape with local traps
        g_norm = np.linalg.norm(genome)
        g_mean = np.mean(genome)
        
        # Multi-modal fitness function (Rastrigin + Griewank blend modified by genome)
        f_base = 100.0 - np.sum((genome[:10] - 0.5) ** 2) * 1.5 - (10.0 * np.sum(1 - np.cos(2 * np.pi * genome[:10])))
        fitness = float(np.clip(f_base - diff * 25.0, 0.0, 100.0))
        
        # Behavioral descriptor b1: Energy consumption / Speed
        b1 = float(np.clip(0.5 + 0.5 * np.sin(g_norm * 0.8 + g_mean * 2.0), 0.0, 1.0))
        # Behavioral descriptor b2: Exploration diversity / Gait symmetry
        b2 = float(np.clip(0.5 + 0.5 * np.cos(np.sum(genome[10:20]) * 0.6), 0.0, 1.0))
        
        return fitness, (b1, b2)

    def add_to_archive(self, individual: Individual) -> bool:
        """
        Inserts individual into archive if cell is unoccupied or fitness > current elite.
        Returns True if newly added/replaced.
        """
        cell = self.behavior_to_cell(individual.behavior)
        current_fit = self.archive_grid[cell]
        
        if individual.fitness > current_fit:
            self.archive_grid[cell] = individual.fitness
            self.archive_individuals[cell] = individual
            return True
        return False

    def mutate_genome(self, genome: np.ndarray) -> np.ndarray:
        """Applies Gaussian mutation with probability mask."""
        mutated = genome.copy()
        mask = np.random.rand(len(genome)) < 0.25
        noise = np.random.normal(0, self.config.mutation_sigma, size=len(genome))
        mutated[mask] += noise[mask]
        return mutated

    def run_map_elites(self) -> QDResult:
        """Executes MAP-Elites Quality-Diversity illumination."""
        # Step 1: Initialize random population
        init_pop_size = self.config.batch_size * 2
        for ind_id in range(init_pop_size):
            genome = np.random.normal(0, 0.5, self.genome_dim)
            fitness, behavior = self.evaluate_genome(genome)
            ind = Individual(ind_id=ind_id, genome=genome, fitness=fitness, behavior=behavior)
            self.add_to_archive(ind)
            
        # Step 2: Main Illumination Loop
        for it in range(self.config.num_iterations):
            # Select random parents from occupied cells
            occupied_cells = list(self.archive_individuals.keys())
            if not occupied_cells:
                continue
                
            for _ in range(self.config.batch_size):
                parent_cell = occupied_cells[np.random.choice(len(occupied_cells))]
                parent = self.archive_individuals[parent_cell]
                
                # Mutate
                child_genome = self.mutate_genome(parent.genome)
                child_fit, child_beh = self.evaluate_genome(child_genome)
                
                child = Individual(
                    ind_id=self.total_evals,
                    genome=child_genome,
                    fitness=child_fit,
                    behavior=child_beh
                )
                self.add_to_archive(child)
                
            # Log QD Metrics
            valid_fits = self.archive_grid[self.archive_grid > -np.inf]
            qd_score = float(np.sum(valid_fits))
            coverage = float(len(valid_fits) / (self.grid_dim * self.grid_dim)) * 100.0
            max_fit = float(np.max(valid_fits)) if len(valid_fits) > 0 else 0.0
            
            self.history_qd_score.append(qd_score)
            self.history_coverage.append(coverage)
            self.history_max_fitness.append(max_fit)
            
        # Find global best individual
        best_cell = max(self.archive_individuals.keys(), key=lambda c: self.archive_individuals[c].fitness)
        best_ind = self.archive_individuals[best_cell]
        
        return QDResult(
            archive_grid=self.archive_grid,
            archive_individuals=self.archive_individuals,
            qd_score_history=self.history_qd_score,
            coverage_history=self.history_coverage,
            max_fitness_history=self.history_max_fitness,
            transfer_matrix=np.zeros((1, 1)),
            active_envs=[],
            total_evaluations=self.total_evals,
            best_individual=best_ind
        )


# ---------------------------------------------------------------------------
# POET (Paired Open-Ended Trailblazer) Engine
# ---------------------------------------------------------------------------

class POETEngine:
    """
    Paired Open-Ended Trailblazer.
    Co-evolves environments and agent policies, transferring policies across niches.
    """
    def __init__(self, config: QDConfig, obs_dim: int = 8, act_dim: int = 2):
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        np.random.seed(config.seed)
        
        self.map_elites = MAPElitesEngine(config, obs_dim, act_dim)
        
        # Initialize root environment and root agent
        root_env = EnvironmentNiche(env_id=0, roughness=0.1, gap_width=0.1, obstacle_density=0.1)
        root_genome = np.random.normal(0, 0.4, self.map_elites.genome_dim)
        root_fit, root_beh = self.map_elites.evaluate_genome(root_genome, root_env)
        root_agent = Individual(ind_id=0, genome=root_genome, fitness=root_fit, behavior=root_beh, env_id=0)
        
        self.active_envs: List[EnvironmentNiche] = [root_env]
        self.paired_agents: Dict[int, Individual] = {0: root_agent}
        self.env_counter = 1
        self.transfer_matrix: np.ndarray = np.zeros((1, 1))

    def run_poet(self) -> QDResult:
        """Executes POET co-evolution with environmental creation and cross-transfers."""
        # First run MAP-Elites baseline
        base_res = self.map_elites.run_map_elites()
        
        for it in range(self.config.num_iterations):
            # 1. Mutate and optimize agents within their assigned environments
            for env in self.active_envs:
                curr_agent = self.paired_agents[env.env_id]
                for _ in range(8):
                    child_g = self.map_elites.mutate_genome(curr_agent.genome)
                    child_fit, child_beh = self.map_elites.evaluate_genome(child_g, env)
                    if child_fit > curr_agent.fitness:
                        curr_agent = Individual(
                            ind_id=self.map_elites.total_evals,
                            genome=child_g,
                            fitness=child_fit,
                            behavior=child_beh,
                            env_id=env.env_id
                        )
                        self.paired_agents[env.env_id] = curr_agent
                        self.map_elites.add_to_archive(curr_agent)
            
            # 2. Procedural Environment Creation (Spawn daughters)
            if len(self.active_envs) < self.config.poet_max_envs and (it % 4 == 0):
                parent_env = self.active_envs[np.random.choice(len(self.active_envs))]
                new_env = parent_env.mutate(self.env_counter)
                
                # Eligibility check: test current parent agent in new env
                test_fit, test_beh = self.map_elites.evaluate_genome(self.paired_agents[parent_env.env_id].genome, new_env)
                # Eligible if not trivial (>20) and not impossible (<85)
                if 20.0 <= test_fit <= 85.0:
                    self.active_envs.append(new_env)
                    self.paired_agents[new_env.env_id] = Individual(
                        ind_id=self.map_elites.total_evals,
                        genome=self.paired_agents[parent_env.env_id].genome.copy(),
                        fitness=test_fit,
                        behavior=test_beh,
                        env_id=new_env.env_id
                    )
                    self.env_counter += 1
            
            # 3. Direct Policy Cross-Transfer Testing
            if it % self.config.transfer_interval == 0 and len(self.active_envs) > 1:
                n_envs = len(self.active_envs)
                self.transfer_matrix = np.zeros((n_envs, n_envs))
                
                for src_idx, src_env in enumerate(self.active_envs):
                    src_agent = self.paired_agents[src_env.env_id]
                    for tgt_idx, tgt_env in enumerate(self.active_envs):
                        tgt_fit, tgt_beh = self.map_elites.evaluate_genome(src_agent.genome, tgt_env)
                        self.transfer_matrix[src_idx, tgt_idx] = tgt_fit
                        
                        # If transfer outperforms existing paired agent in target env, replace!
                        if tgt_fit > self.paired_agents[tgt_env.env_id].fitness:
                            self.paired_agents[tgt_env.env_id] = Individual(
                                ind_id=self.map_elites.total_evals,
                                genome=src_agent.genome.copy(),
                                fitness=tgt_fit,
                                behavior=tgt_beh,
                                env_id=tgt_env.env_id
                            )
                            
        # Final result integration
        best_agent = max(self.paired_agents.values(), key=lambda a: a.fitness)
        
        return QDResult(
            archive_grid=self.map_elites.archive_grid,
            archive_individuals=self.map_elites.archive_individuals,
            qd_score_history=self.map_elites.history_qd_score,
            coverage_history=self.map_elites.history_coverage,
            max_fitness_history=self.map_elites.history_max_fitness,
            transfer_matrix=self.transfer_matrix,
            active_envs=self.active_envs,
            total_evaluations=self.map_elites.total_evals,
            best_individual=best_agent
        )
