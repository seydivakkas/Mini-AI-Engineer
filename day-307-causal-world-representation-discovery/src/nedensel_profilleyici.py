"""
Day 307: Causal Discovery & Do-Calculus Profiler.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any
import numpy as np
from .nedensel_dunya_motoru import CausalDiscoveryResult


class CausalProfiler:
    """
    Profiles Structural Causal Model discovery, interventional generalization,
    and acyclicity adherence.
    """
    
    @staticmethod
    def profile_results(result: CausalDiscoveryResult) -> Dict[str, Any]:
        """
        Generates diagnostic report on causal discovery performance.
        """
        acyclicity_status = "STRICT_DAG_SATISFIED" if result.structural_hamming_distance <= 3 else "PARTIALLY_ACYCLIC"
        causal_generalization_status = "ROBUST_INTERVENTIONAL" if result.interventional_mse < 0.25 else "CORRELATION_LIMITED"
        
        return {
            "structural_hamming_distance": result.structural_hamming_distance,
            "dag_true_positive_rate_pct": round(result.dag_true_positive_rate_pct, 2),
            "dag_false_discovery_rate_pct": round(result.dag_false_discovery_rate_pct, 2),
            "interventional_mse": round(result.interventional_mse, 4),
            "counterfactual_mse": round(result.counterfactual_mse, 4),
            "reconstruction_mse": round(result.reconstruction_mse, 4),
            "acyclicity_status": acyclicity_status,
            "causal_generalization_status": causal_generalization_status
        }
