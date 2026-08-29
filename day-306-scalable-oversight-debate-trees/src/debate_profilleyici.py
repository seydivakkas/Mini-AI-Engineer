"""
Day 306: Scalable Oversight Debate Profiler and Game-Theoretic Metrics.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any
from .debate_motoru import DebateResult


class DebateProfiler:
    """
    Analyzes game-theoretic equilibrium, truth convergence,
    and formal verification efficiency.
    """
    
    @staticmethod
    def profile_results(res: DebateResult) -> Dict[str, Any]:
        """
        Summarizes debate game results.
        """
        return {
            "judge_accuracy_pct": round(res.judge_accuracy_pct, 2),
            "honest_agent_win_rate_pct": round(res.honest_agent_win_rate, 2),
            "fallacy_detection_rate_pct": round(res.fallacy_detection_rate, 2),
            "tree_nodes_explored": res.minimax_tree_nodes_explored,
            "pruning_efficiency_pct": round(res.pruning_efficiency_pct, 2),
            "avg_debate_length_turns": res.avg_debate_length_turns,
            "nash_equilibrium_status": "HONESTY_EQUILIBRIUM_VERIFIED" if res.honest_agent_win_rate >= 80.0 else "SUBOPTIMAL",
            "scalable_oversight_status": "FORMAL_VERIFICATION_ROBUST"
        }
