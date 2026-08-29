"""
Day 311: Automated Scientific Discovery Profiler.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any
from .bilimsel_kesif_motoru import ScientificDiscoveryResult


class ScientificDiscoveryProfiler:
    """
    Profiles equation precision, parameter error, parsimony BIC, and extrapolation.
    """
    
    @staticmethod
    def profile_results(result: ScientificDiscoveryResult) -> Dict[str, Any]:
        """
        Generates telemetry profile dictionary.
        """
        tier = "EXACT_PHYSICAL_LAW_DISCOVERED" if result.equation_recovery_precision_pct >= 95.0 else "APPROXIMATE_EMPIRICAL_FIT"
        
        return {
            "equation_recovery_precision_pct": round(result.equation_recovery_precision_pct, 2),
            "avg_parameter_relative_error_pct": round(result.avg_parameter_relative_error_pct, 2),
            "ood_extrapolation_r2": round(result.ood_extrapolation_r2, 4),
            "parsimony_bic_score": round(result.parsimony_bic_score, 2),
            "num_discovered_equations": len(result.discovered_equations),
            "discovery_tier": tier
        }
