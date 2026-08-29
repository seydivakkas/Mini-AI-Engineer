"""
Day 304: Weak-to-Strong Superalignment Profiler and Calibration Metrics.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any
from .superalignment_motoru import SuperalignmentResult


class SuperalignmentProfiler:
    """
    Computes Superalignment diagnostics, PGR score,
    calibration improvement, and conformal coverage bounds.
    """
    
    @staticmethod
    def profile_results(res: SuperalignmentResult) -> Dict[str, Any]:
        """
        Generates a structured diagnostic summary.
        """
        delta_acc = res.weak_to_strong_acc - res.weak_acc
        ece_improvement = max(0.0, res.ece_before - res.ece_after)
        
        return {
            "weak_supervisor_acc": round(res.weak_acc, 2),
            "strong_ceiling_acc": round(res.strong_ceiling_acc, 2),
            "weak_to_strong_acc": round(res.weak_to_strong_acc, 2),
            "generalization_delta": round(delta_acc, 2),
            "pgr_score_pct": round(res.pgr_score, 2),
            "calibrated_temperature": round(res.temperature, 3),
            "ece_before_pct": round(res.ece_before, 2),
            "ece_after_pct": round(res.ece_after, 2),
            "ece_reduction_pct": round(ece_improvement, 2),
            "conformal_coverage_pct": round(res.conformal_coverage_pct, 2),
            "target_coverage_pct": 90.0,
            "avg_prediction_set_size": round(res.avg_conformal_set_size, 2),
            "superalignment_status": "SUPERIOR_GENERALIZATION_ACHIEVED" if res.pgr_score > 20.0 else "BASELINE"
        }
