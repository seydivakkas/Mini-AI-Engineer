"""
Day 308: Polymath Agent & Skill Graph Profiler.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any
from .polymath_motoru import PolymathResult


class PolymathProfiler:
    """
    Profiles recursive skill synthesis, self-reflection recovery, and graph growth.
    """
    
    @staticmethod
    def profile_results(result: PolymathResult) -> Dict[str, Any]:
        """
        Generates diagnostic metric summary.
        """
        autonomy_tier = "POLYMATH_RECURSIVE_SYNTHESIS_VERIFIED" if result.skill_synthesis_success_rate_pct >= 95.0 else "SUBOPTIMAL"
        
        return {
            "skill_synthesis_success_rate_pct": round(result.skill_synthesis_success_rate_pct, 2),
            "cross_domain_reuse_efficiency_pct": round(result.cross_domain_reuse_efficiency_pct, 2),
            "reflection_error_recovery_rate_pct": round(result.reflection_error_recovery_rate_pct, 2),
            "total_skills_synthesized": result.total_skills_synthesized,
            "memory_graph_density": round(result.memory_graph_density, 4),
            "avg_execution_latency_ms": round(result.avg_execution_latency_ms, 2),
            "autonomy_tier": autonomy_tier
        }
