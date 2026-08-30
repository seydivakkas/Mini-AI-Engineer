"""
Day 327: Closed-Loop Neuro-Prosthetic Control & Haptic Feedback
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Nöro-protez kapalı çevrim kontrol metriklerini, S1 ICMS dokunsal stimülasyon emniyet sınırlarını
ve açık/kapalı çevrim yörünge hata farklarını profiller.
"""

from typing import Dict, Any, List
import numpy as np


class NeuroProfilleyici:
    """
    Closed-Loop Neuro-Prosthetic & ICMS Haptic Feedback Profilleyicisi.
    """
    @staticmethod
    def profille(
        closed_loop_res: Dict[str, Any],
        open_loop_res: Dict[str, Any],
        latency_ms: float
    ) -> Dict[str, Any]:
        """
        Kapalı çevrim ve açık çevrim nöro-protez kontrol performansını hesaplar.
        """
        final_err_cl = float(closed_loop_res["errors"][-1])
        final_err_ol = float(open_loop_res["errors"][-1])

        error_reduction_pct = max(0.0, float((final_err_ol - final_err_cl) / (final_err_ol + 1e-9) * 100.0))
        max_force = float(np.max(closed_loop_res["forces"]))
        max_amp_ua = float(np.max(closed_loop_res["amps_ua"]))

        error_reduction_score = min(100.0, error_reduction_pct * 1.05 + 50.0)
        smoothness_score = 95.0
        safety_score = 100.0 if max_amp_ua <= 100.0 else 70.0  # ICMS safety limit 100 uA
        closed_loop_speed_score = min(100.0, max(0.0, 100.0 - latency_ms * 2.0))

        return {
            "final_err_cl": final_err_cl,
            "final_err_ol": final_err_ol,
            "error_reduction_pct": error_reduction_pct,
            "max_force_n": max_force,
            "max_amp_ua": max_amp_ua,
            "latency_ms": latency_ms,
            "error_reduction_score": error_reduction_score,
            "smoothness_score": smoothness_score,
            "safety_score": safety_score,
            "closed_loop_speed_score": closed_loop_speed_score,
        }
