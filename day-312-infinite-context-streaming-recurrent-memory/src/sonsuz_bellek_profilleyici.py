"""
Day 312: Infinite Context Streaming Memory Profiler.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any
from .sonsuz_bellek_motoru import StreamingMemoryResult


class StreamingMemoryProfiler:
    """
    Profiles retention accuracy, constant memory efficiency, and latency scaling.
    """
    
    @staticmethod
    def profile_results(result: StreamingMemoryResult) -> Dict[str, Any]:
        """
        Generates telemetry profile dictionary.
        """
        tier = "OPTIMAL_INFINITE_STREAMING_MEMORY" if result.memory_compression_ratio_pct >= 90.0 else "SUBOPTIMAL_MEMORY"
        speedup = result.quadratic_kv_latency_ms / max(result.avg_step_latency_ms, 1e-4)
        
        return {
            "retrieval_accuracy_pct": round(result.retrieval_accuracy_pct, 2),
            "context_retention_index": round(result.context_retention_index, 4),
            "memory_compression_ratio_pct": round(result.memory_compression_ratio_pct, 2),
            "avg_step_latency_ms": round(result.avg_step_latency_ms, 4),
            "kv_cache_speedup_factor": round(speedup, 1),
            "stream_length_tokens": result.stream_length,
            "memory_tier": tier
        }
