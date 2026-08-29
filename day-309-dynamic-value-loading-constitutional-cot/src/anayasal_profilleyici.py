"""
Day 309: Constitutional CoT & Value Steering Profiler.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any
from .anayasal_cot_motoru import ConstitutionalResult


class ConstitutionalProfiler:
    """
    Profiles value alignment, adversarial suppression, and Pareto trade-offs.
    """
    
    @staticmethod
    def profile_results(result: ConstitutionalResult) -> Dict[str, Any]:
        """
        Generates diagnostic report on Constitutional CoT performance.
        """
        alignment_tier = "CONSTITUTIONAL_SUPER_ALIGNMENT_ACTIVE" if result.violation_suppression_rate_pct >= 90.0 else "PARTIALLY_ALIGNED"
        
        return {
            "value_alignment_score_pct": round(result.value_alignment_score_pct, 2),
            "violation_suppression_rate_pct": round(result.violation_suppression_rate_pct, 2),
            "helpfulness_retention_pct": round(result.helpfulness_retention_pct, 2),
            "unsteered_violation_rate_pct": round(result.unsteered_violation_rate_pct, 2),
            "steered_violation_rate_pct": round(result.steered_violation_rate_pct, 2),
            "avg_cot_steps_to_resolution": round(result.avg_cot_steps_to_resolution, 1),
            "alignment_tier": alignment_tier
        }
