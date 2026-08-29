"""
Day 316: Byzantine Fault Tolerance Profiler.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any
from .bizans_hata_toleransi import ByzantineBenchmarkResult


class ByzantineDefenseProfiler:
    """
    Profiles Byzantine attack mitigation, gradient cosine fidelity, and attacker detection metrics.
    """
    
    @staticmethod
    def profile_results(result: ByzantineBenchmarkResult) -> Dict[str, Any]:
        """
        Summarizes swarm robustness and fault tolerance.
        """
        tier = "HIGH_INTEGRITY_BYZANTINE_RESILIENT" if result.attack_mitigation_ratio_pct >= 85.0 else "VULNERABLE_SWARM"
        
        return {
            "attack_mitigation_ratio_pct": round(result.attack_mitigation_ratio_pct, 2),
            "bulyan_mean_cosine": round(result.mean_cosine_fidelity["Bulyan"], 4),
            "multi_krum_mean_cosine": round(result.mean_cosine_fidelity["Multi-Krum"], 4),
            "trimmed_mean_cosine": round(result.mean_cosine_fidelity["Trimmed-Mean"], 4),
            "naive_mean_cosine": round(result.mean_cosine_fidelity["Naive-Mean"], 4),
            "byzantine_detection_precision_pct": round(result.byzantine_detection_precision_pct, 2),
            "byzantine_detection_recall_pct": round(result.byzantine_detection_recall_pct, 2),
            "resilience_tier": tier
        }
