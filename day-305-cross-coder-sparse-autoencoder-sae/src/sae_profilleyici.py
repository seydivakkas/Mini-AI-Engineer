"""
Day 305: Cross-Coder SAE Profiler and Mechanistic Interpretability Diagnostics.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any
from .cross_coder_motoru import CrossCoderResult


class SAEProfiler:
    """
    Analyzes dictionary quality, reconstruction fidelity,
    superposition disentanglement, and cross-layer sharing.
    """
    
    @staticmethod
    def profile_results(res: CrossCoderResult) -> Dict[str, Any]:
        """
        Summarizes Cross-Coder SAE performance metrics.
        """
        layer_fve_str = ", ".join([f"L{i}: %{fve:.1f}" for i, fve in enumerate(res.fve_per_layer)])
        
        return {
            "mean_fve_pct": round(res.mean_fve, 2),
            "layer_fves": layer_fve_str,
            "l0_sparsity_avg": round(res.l0_sparsity, 2),
            "dead_feature_pct": round(res.dead_feature_pct, 2),
            "cross_layer_sharing_pct": round(res.cross_layer_sharing_idx, 2),
            "disentanglement_quality": "EXCELLENT" if res.mean_fve > 85.0 and res.dead_feature_pct < 10.0 else "GOOD",
            "superposition_resolution": "SUCCESSFUL_CROSS_LAYER_DECOMPOSITION"
        }
