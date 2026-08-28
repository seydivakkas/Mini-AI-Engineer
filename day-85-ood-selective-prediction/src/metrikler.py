"""
OOD Tespiti ve Değerlendirme Metrikleri
---------------------------------------
AUROC (Area Under ROC Curve), AUPR (Area Under Precision-Recall),
FPR95 (False Positive Rate at 95% TPR) metrik hesaplayıcıları.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score


class OODMetrikleri:
    """
    Dağılım Dışı (OOD) ayrıştırma kalitesini ölçen metrik motoru.
    """
    @staticmethod
    def hesapla_ood_metrikleri(
        id_skorlar: np.ndarray,
        ood_skorlar: np.ndarray
    ) -> Dict[str, Any]:
        """
        ID (pozitif = 1) ve OOD (negatif = 0) skorları üzerinden AUROC, AUPR ve FPR95 hesaplar.
        """
        id_skorlar = np.array(id_skorlar).flatten()
        ood_skorlar = np.array(ood_skorlar).flatten()

        y_true = np.concatenate([np.ones_like(id_skorlar), np.zeros_like(ood_skorlar)])
        y_scores = np.concatenate([id_skorlar, ood_skorlar])

        # AUROC
        auroc = roc_auc_score(y_true, y_scores) * 100.0

        # AUPR (In-distribution positive)
        aupr = average_precision_score(y_true, y_scores) * 100.0

        # ROC Eğrisi
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        # FPR95: TPR >= 0.95 olduğu noktadaki minimum FPR
        # TPR >= 0.95 olan ilk indeks
        idx_tpr95 = np.where(tpr >= 0.95)[0]
        if len(idx_tpr95) > 0:
            fpr95 = fpr[idx_tpr95[0]] * 100.0
            esik_tpr95 = thresholds[idx_tpr95[0]]
        else:
            fpr95 = 100.0
            esik_tpr95 = thresholds[0]

        return {
            "auroc": auroc,
            "aupr": aupr,
            "fpr95": fpr95,
            "esik_tpr95": esik_tpr95,
            "fpr_dizisi": fpr,
            "tpr_dizisi": tpr
        }
