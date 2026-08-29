"""
Day 314: Game-Theoretic Mechanism Telemetry Profiler.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any
from .oyun_teorisi_mekanizma import MechanismResult


class GameTheoreticProfiler:
    """
    Profiles social welfare efficiency, DSIC incentive stability, and Nash surplus gains.
    """
    
    @staticmethod
    def profile_results(result: MechanismResult) -> Dict[str, Any]:
        """
        Summarizes game-theoretic equilibrium metrics.
        """
        tier = "PARETO_OPTIMAL_DSIC_STABLE" if result.pareto_efficiency_pct >= 99.0 else "SUBOPTIMAL_EQUILIBRIUM"
        total_payment = sum(result.vcg_payments.values())
        
        return {
            "vcg_optimal_outcome": result.vcg_optimal_outcome,
            "vcg_social_welfare": round(result.vcg_social_welfare, 2),
            "vcg_total_payments": round(total_payment, 2),
            "dsic_truthful_stability_gain": round(result.truthful_vs_manipulated_utility_gain, 4),
            "nash_bargaining_product": round(result.total_nash_product, 4),
            "pareto_efficiency_pct": round(result.pareto_efficiency_pct, 2),
            "equilibrium_tier": tier
        }
