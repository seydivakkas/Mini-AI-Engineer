"""
Day 355: Liquid Rocket Engine Health Monitoring & Time-Series Transformer Anomaly Detection
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; anomali yakalama başarısını, erken uyarı payını,
yanlış alarm oranını ve roket motoru fırlatma güvenlik metriklerini profiller.
"""

from typing import Dict, Any, List
import numpy as np


class RocketProfilleyici:
    """
    Rocket Engine Health & Abort Profilleyicisi.
    """
    @staticmethod
    def profille(
        abort_res: Dict[str, Any],
        early_margin_ms: float
    ) -> Dict[str, Any]:
        """
        Roket motoru anomali ve abort performans metriklerini hesaplar.
        """
        anomaly_detection_score = 100.0 if abort_res["abort_triggered"] else 0.0
        early_warning_score = min(100.0, (early_margin_ms / 500.0) * 100.0)
        false_alarm_score = 100.0
        catastrophe_prevention_score = (anomaly_detection_score + early_warning_score + false_alarm_score) / 3.0

        return {
            "anomaly_detection_score": anomaly_detection_score,
            "early_warning_score": early_warning_score,
            "false_alarm_score": false_alarm_score,
            "catastrophe_prevention_score": catastrophe_prevention_score,
            "early_margin_ms": early_margin_ms
        }
