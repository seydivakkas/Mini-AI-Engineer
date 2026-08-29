"""
Day 319: Free Energy Principle & Active Inference Profiler.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any
import numpy as np
from .serbest_enerji_aktif_cikarim import FEPSimulationResult


class FEPProfiler:
    """
    Profiles Variational Free Energy, Epistemic Curiosity vs Pragmatic Exploitation, and entropy reduction.
    """
    
    @staticmethod
    def profile_results(result: FEPSimulationResult) -> Dict[str, Any]:
        """
        Summarizes FEP active inference metrics.
        """
        init_entropy = result.state_entropy_history[0] if result.state_entropy_history else 1.0
        final_entropy = result.state_entropy_history[-1] if result.state_entropy_history else 0.0
        entropy_reduction_pct = max(0.0, float((init_entropy - final_entropy) / (init_entropy + 1e-8) * 100.0))
        
        tier = "OPTIMAL_ACTIVE_INFERENCE_AGENT" if result.goal_reached and result.total_epistemic_gain > 0.1 else "SUBOPTIMAL_EXPLORATION"
        
        return {
            "goal_reached": result.goal_reached,
            "trajectory_steps": len(result.trajectory_actions),
            "total_epistemic_gain": round(result.total_epistemic_gain, 4),
            "final_variational_free_energy": round(result.final_vfe, 4),
            "entropy_reduction_pct": round(entropy_reduction_pct, 2),
            "initial_state_entropy": round(init_entropy, 4),
            "final_state_entropy": round(final_entropy, 4),
            "fep_agent_tier": tier
        }
