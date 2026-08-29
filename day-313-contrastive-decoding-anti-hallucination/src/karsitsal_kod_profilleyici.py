"""
Day 313: Contrastive Decoding Telemetry Profiler.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any
from .karsitsal_kod_cozucu import ContrastiveDecodingResult


class ContrastiveDecodingProfiler:
    """
    Profiles factuality gain, hallucination suppression, and calibration metrics.
    """
    
    @staticmethod
    def profile_results(result: ContrastiveDecodingResult) -> Dict[str, Any]:
        """
        Summarizes factual calibration performance.
        """
        tier = "HIGH_PRECISION_GROUNDED" if result.contrastive_factuality_pct >= 90.0 else "STANDARD_GROUNDED"
        
        return {
            "standard_factuality_pct": round(result.standard_factuality_pct, 2),
            "contrastive_factuality_pct": round(result.contrastive_factuality_pct, 2),
            "hallucination_reduction_pct": round(result.hallucination_reduction_pct, 2),
            "standard_ece": round(result.standard_ece, 4),
            "contrastive_ece": round(result.contrastive_ece, 4),
            "total_tokens_evaluated": result.tokens_generated,
            "calibration_tier": tier
        }
