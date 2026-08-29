"""
Day 318: Neuro-Symbolic Logic & Differentiable Theorem Profiler.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any
import numpy as np
from .noro_sembolik_mantik import NeuroSymbolicResult


class NeuroSymbolicProfiler:
    """
    Profiles differentiable First-Order Logic convergence, axiom satisfaction, and theorem proof accuracy.
    """
    
    @staticmethod
    def profile_results(result: NeuroSymbolicResult) -> Dict[str, Any]:
        """
        Summarizes neuro-symbolic logic metrics.
        """
        mean_axiom_sat = float(np.mean(list(result.rule_satisfaction_rates.values())) * 100.0)
        tier = "LOGIC_GROUNDED_NEURO_SYMBOLIC" if result.theorem_proof_accuracy_pct >= 75.0 and mean_axiom_sat >= 70.0 else "UNCONSTRAINED_NEURAL"
        
        return {
            "t_norm_framework": result.t_norm_name.upper(),
            "theorem_proof_accuracy_pct": round(result.theorem_proof_accuracy_pct, 2),
            "mean_axiom_satisfaction_pct": round(mean_axiom_sat, 2),
            "final_total_loss": round(result.total_loss, 4),
            "logical_violation_loss": round(result.final_logical_violation_loss, 4),
            "axiom_1_base_sat": round(result.rule_satisfaction_rates.get("Axiom_1_Base", 0.0), 4),
            "axiom_2_transitivity_sat": round(result.rule_satisfaction_rates.get("Axiom_2_Transitivity", 0.0), 4),
            "axiom_3_asymmetry_sat": round(result.rule_satisfaction_rates.get("Axiom_3_Asymmetry", 0.0), 4),
            "neuro_symbolic_tier": tier
        }
