"""
Day 302: Meta-NAS Profiler and Benchmark Metrics.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, List, Any
import numpy as np
from .meta_nas_motoru import NASSearchResult, ArchitectureCandidate


class MetaNASProfiler:
    """
    Computes architectural search diagnostics, Pareto hypervolume,
    discretization gap, and hardware acceleration metrics.
    """
    
    @staticmethod
    def compute_hypervolume(pareto_frontier: List[ArchitectureCandidate], 
                            ref_point: tuple = (0.0, 50.0, 5.0)) -> float:
        """
        Calculates 2D/3D Hypervolume metric bounded by ref_point (min_acc, max_flops, max_lat).
        Normalized score [0, 100].
        """
        if not pareto_frontier:
            return 0.0
            
        # Simplified 2D Hypervolume (Accuracy vs FLOPs)
        sorted_frontier = sorted(pareto_frontier, key=lambda c: c.flops_m)
        min_acc, max_flops, _ = ref_point
        
        hv = 0.0
        prev_flops = 0.0
        for cand in sorted_frontier:
            if cand.flops_m < max_flops:
                width = (max_flops - cand.flops_m)
                height = max(0.0, cand.accuracy - min_acc)
                hv += width * height
                
        # Normalized by max potential area (100% acc * max_flops)
        max_area = 100.0 * max_flops
        return float(min(100.0, (hv / max(1e-5, max_area)) * 100.0))

    @staticmethod
    def profile_search(result: NASSearchResult) -> Dict[str, Any]:
        """
        Generates a comprehensive diagnostic report from search result.
        """
        history = result.search_history
        init_train_loss = history["train_loss"][0] if history["train_loss"] else 0.0
        final_train_loss = history["train_loss"][-1] if history["train_loss"] else 0.0
        init_val_acc = history["val_acc"][0] if history["val_acc"] else 0.0
        final_val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
        
        # Discretization gap
        best_cand = result.best_candidate
        discretization_gap = abs(final_val_acc - best_cand.accuracy)
        
        # FLOPs reduction compared to maximal dense supernet
        max_possible_flops = 50.0  # reference unpruned baseline
        flops_reduction = max(0.0, (1.0 - best_cand.flops_m / max_possible_flops) * 100.0)
        
        # Latency speedup
        baseline_latency = 2.50  # ms
        speedup = baseline_latency / max(0.01, best_cand.latency_ms)
        
        # Hypervolume of Pareto frontier
        hv_score = MetaNASProfiler.compute_hypervolume(result.pareto_frontier)
        
        # Entropy reduction (measuring certainty in architecture selection)
        init_entropy = history["alpha_entropy"][0] if history["alpha_entropy"] else 1.79
        final_entropy = history["alpha_entropy"][-1] if history["alpha_entropy"] else 0.0
        entropy_reduction = max(0.0, (init_entropy - final_entropy) / max(1e-5, init_entropy) * 100.0)
        
        return {
            "initial_train_loss": round(init_train_loss, 4),
            "final_train_loss": round(final_train_loss, 4),
            "initial_val_acc": round(init_val_acc, 2),
            "final_val_acc": round(final_val_acc, 2),
            "best_cand_acc": round(best_cand.accuracy, 2),
            "best_cand_flops_m": round(best_cand.flops_m, 4),
            "best_cand_latency_ms": round(best_cand.latency_ms, 3),
            "discretization_gap": round(discretization_gap, 2),
            "flops_reduction_pct": round(flops_reduction, 1),
            "latency_speedup_x": round(speedup, 2),
            "pareto_hypervolume_score": round(hv_score, 2),
            "entropy_reduction_pct": round(entropy_reduction, 1),
            "pareto_frontier_count": len(result.pareto_frontier),
            "total_candidates_evaluated": len(result.all_candidates),
            "search_time_sec": round(result.search_time_sec, 3)
        }
