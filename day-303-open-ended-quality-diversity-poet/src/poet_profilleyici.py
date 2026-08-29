"""
Day 303: Quality-Diversity & POET Diagnostic Profiler.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, List
import numpy as np
from .map_elites_poet_motoru import QDResult


class POETProfiler:
    """
    Computes Quality-Diversity metrics, archive coverage,
    environmental niche complexity, and cross-transfer efficacy.
    """
    
    @staticmethod
    def profile_results(result: QDResult) -> Dict[str, Any]:
        """
        Extracts comprehensive metrics from QD/POET co-evolution.
        """
        valid_fits = result.archive_grid[result.archive_grid > -np.inf]
        grid_total = result.archive_grid.size
        
        qd_score = float(np.sum(valid_fits)) if len(valid_fits) > 0 else 0.0
        coverage_pct = float(len(valid_fits) / grid_total) * 100.0
        max_fitness = float(np.max(valid_fits)) if len(valid_fits) > 0 else 0.0
        mean_fitness = float(np.mean(valid_fits)) if len(valid_fits) > 0 else 0.0
        
        # Environmental diversity
        envs = result.active_envs
        env_count = len(envs)
        avg_roughness = float(np.mean([e.roughness for e in envs])) if envs else 0.0
        avg_gap = float(np.mean([e.gap_width for e in envs])) if envs else 0.0
        avg_obstacle = float(np.mean([e.obstacle_density for e in envs])) if envs else 0.0
        
        # Transfer efficacy
        mat = result.transfer_matrix
        transfer_success_count = 0
        transfer_eval_count = 0
        if mat.ndim == 2 and mat.shape[0] > 1:
            n = mat.shape[0]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        transfer_eval_count += 1
                        # If cross transfer achieves reasonable fitness
                        if mat[i, j] > 30.0:
                            transfer_success_count += 1
                            
        transfer_rate = (transfer_success_count / max(1, transfer_eval_count)) * 100.0
        
        return {
            "qd_score": round(qd_score, 2),
            "archive_coverage_pct": round(coverage_pct, 2),
            "max_elite_fitness": round(max_fitness, 2),
            "mean_elite_fitness": round(mean_fitness, 2),
            "total_occupied_niches": len(valid_fits),
            "total_grid_capacity": grid_total,
            "active_environments_count": env_count,
            "avg_env_roughness": round(avg_roughness, 3),
            "avg_env_gap_width": round(avg_gap, 3),
            "avg_env_obstacle_density": round(avg_obstacle, 3),
            "cross_transfer_success_rate": round(transfer_rate, 1),
            "total_evaluations": result.total_evaluations,
            "best_individual_id": result.best_individual.ind_id,
            "best_individual_fitness": round(result.best_individual.fitness, 2)
        }
