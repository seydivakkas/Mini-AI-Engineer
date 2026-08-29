"""
Day 310: Diffusion-Based Latent Planner Profiler.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any
from .difuzyon_planlayici_motoru import DiffusionPlannerResult


class DiffusionPlannerProfiler:
    """
    Profiles goal reachability, obstacle clearance, smoothness, and sampling acceleration.
    """
    
    @staticmethod
    def profile_results(result: DiffusionPlannerResult) -> Dict[str, Any]:
        """
        Generates telemetry profile dictionary.
        """
        tier = "OPTIMAL_CONTINUOUS_DIFFUSION_PLANNER" if result.goal_reachability_rate_pct >= 85.0 else "SUBOPTIMAL_PLANNER"
        
        return {
            "goal_reachability_rate_pct": round(result.goal_reachability_rate_pct, 2),
            "obstacle_avoidance_rate_pct": round(result.obstacle_avoidance_rate_pct, 2),
            "trajectory_smoothness_score": round(result.trajectory_smoothness_score, 2),
            "ddim_speedup_factor": round(result.ddim_speedup_factor, 1),
            "avg_trajectory_length": round(result.avg_trajectory_length, 2),
            "planner_tier": tier
        }
