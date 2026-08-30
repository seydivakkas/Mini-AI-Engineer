"""
Day 337: Non-Invasive BCI P300 Speller & Error-Related Potential (ErrP) Real-Time Correction
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; ham BCI yazma doğruluğunu, ErrP düzeltmeli doğruluğu,
Bilgi Transfer Hızını (ITR) ve BCI sistem hazır bulunurluk metriklerini profiller.
"""

from typing import Dict, Any, List
import numpy as np


class P300Profilleyici:
    """
    Non-Invasive BCI P300 Speller Profilleyicisi.
    """
    @staticmethod
    def profille(
        raw_accuracy: float,
        corrected_accuracy: float,
        itr_bits_per_min: float,
        itr_history: List[float] = None
    ) -> Dict[str, Any]:
        """
        BCI P300 Speller ve ErrP otomatik düzeltme skorlarını hesaplar.
        """
        if itr_history is None:
            itr_history = [15, 22, 28, 35, 42, 48, 52, 55, 58, float(itr_bits_per_min)]

        p300_detection_score = float(raw_accuracy)
        errp_correction_score = float(corrected_accuracy)
        itr_score = min(100.0, float(itr_bits_per_min) * 1.6)
        bci_readiness_score = (p300_detection_score + errp_correction_score + itr_score) / 3.0

        return {
            "raw_accuracy": raw_accuracy,
            "corrected_accuracy": corrected_accuracy,
            "itr_bits_per_min": itr_bits_per_min,
            "itr_history": itr_history,
            "p300_detection_score": p300_detection_score,
            "errp_correction_score": errp_correction_score,
            "itr_score": itr_score,
            "bci_readiness_score": bci_readiness_score,
        }
