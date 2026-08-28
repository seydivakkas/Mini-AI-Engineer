"""
Model Değerlendirme ve İstatistiksel Metrik Hesaplayıcı (ROC-AUC, PR-Curve, Confusion Matrix).
"""

from typing import Dict, Any, Tuple
import numpy as np


class MetrikHesaplayici:
    """Sınıflandırma modelleri için Confusion Matrix, ROC-AUC ve Precision-Recall eğrilerini hesaplar."""

    @classmethod
    def karmasiklik_matrisi_hesapla(
        cls,
        y_gercek: np.ndarray,
        y_olasilik: np.ndarray,
        esik: float = 0.50
    ) -> Dict[str, Any]:
        """Belirtilen olasılık eşiğine göre Karmaşıklık Matrisi (Confusion Matrix) ve türetilmiş metrikleri üretir."""
        y_true = np.asarray(y_gercek, dtype=int)
        y_pred = (np.asarray(y_olasilik, dtype=float) >= esik).astype(int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        toplam = max(len(y_true), 1)
        dogruluk = (tp + tn) / toplam
        kesinlik = tp / max(tp + fp, 1)
        duyarlilik = tp / max(tp + fn, 1)
        ozgulluk = tn / max(tn + fp, 1)
        f1 = (2.0 * kesinlik * duyarlilik) / max(kesinlik + duyarlilik, 1e-8)

        # Matthews Correlation Coefficient (MCC)
        pay = (tp * tn) - (fp * fn)
        payda = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) + 1e-12
        mcc = pay / payda

        return {
            "esik": esik,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "matris": [[tn, fp], [fn, tp]],
            "dogruluk_acc": float(round(dogruluk * 100.0, 2)),
            "kesinlik_precision": float(round(kesinlik * 100.0, 2)),
            "duyarlilik_recall": float(round(duyarlilik * 100.0, 2)),
            "ozgulluk_specificity": float(round(ozgulluk * 100.0, 2)),
            "f1_skoru": float(round(f1, 4)),
            "mcc_skoru": float(round(mcc, 4))
        }

    @classmethod
    def roc_egrisi_hesapla(
        cls,
        y_gercek: np.ndarray,
        y_olasilik: np.ndarray,
        esik_sayisi: int = 100
    ) -> Dict[str, Any]:
        """Tüm eşik değerleri üzerinde TPR (True Positive Rate) ve FPR (False Positive Rate) hesaplar."""
        y_true = np.asarray(y_gercek, dtype=int)
        y_prob = np.asarray(y_olasilik, dtype=float)

        esikler = np.linspace(1.0, 0.0, esik_sayisi)
        tpr_list = []
        fpr_list = []

        n_pos = max(int(np.sum(y_true == 1)), 1)
        n_neg = max(int(np.sum(y_true == 0)), 1)

        for th in esikler:
            pred = (y_prob >= th).astype(int)
            tp = np.sum((y_true == 1) & (pred == 1))
            fp = np.sum((y_true == 0) & (pred == 1))
            tpr_list.append(float(tp / n_pos))
            fpr_list.append(float(fp / n_neg))

        tpr_arr = np.array(tpr_list)
        fpr_arr = np.array(fpr_list)

        # Trapezoidal Entegrasyon ile ROC-AUC Hesabı
        if hasattr(np, "trapezoid"):
            auc = float(np.trapezoid(tpr_arr, fpr_arr))
        else:
            auc = float(np.trapz(tpr_arr, fpr_arr))

        return {
            "roc_auc": float(round(auc, 4)),
            "fpr": fpr_arr,
            "tpr": tpr_arr,
            "esikler": esikler
        }

    @classmethod
    def pr_egrisi_hesapla(
        cls,
        y_gercek: np.ndarray,
        y_olasilik: np.ndarray,
        esik_sayisi: int = 100
    ) -> Dict[str, Any]:
        """Precision-Recall eğrisi ve Average Precision (AP) skorunu hesaplar."""
        y_true = np.asarray(y_gercek, dtype=int)
        y_prob = np.asarray(y_olasilik, dtype=float)

        esikler = np.linspace(1.0, 0.0, esik_sayisi)
        precision_list = []
        recall_list = []

        n_pos = max(int(np.sum(y_true == 1)), 1)

        for th in esikler:
            pred = (y_prob >= th).astype(int)
            tp = np.sum((y_true == 1) & (pred == 1))
            fp = np.sum((y_true == 0) & (pred == 1))

            p = tp / max(tp + fp, 1e-8)
            r = tp / n_pos
            precision_list.append(float(p))
            recall_list.append(float(r))

        p_arr = np.array(precision_list)
        r_arr = np.array(recall_list)

        # Sıralı integral ile Average Precision
        if hasattr(np, "trapezoid"):
            ap = float(np.trapezoid(p_arr, r_arr))
        else:
            ap = float(np.trapz(p_arr, r_arr))

        return {
            "average_precision_ap": float(round(ap, 4)),
            "taban_oran": float(round(n_pos / len(y_true), 4)),
            "precision": p_arr,
            "recall": r_arr,
            "esikler": esikler
        }
