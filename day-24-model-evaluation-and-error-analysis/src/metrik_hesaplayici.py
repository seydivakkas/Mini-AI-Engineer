"""Model Değerlendirme ve Çok Sınıflı Metrik Hesaplayıcı Modülü.

Bu modül; Çok sınıflı ROC-AUC (One-vs-Rest), PR-AUC (Average Precision),
Top-k doğruluğu ve sınıf bazında detaylı performans metriklerini hesaplar.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


class MetrikHesaplayici:
    """Çok sınıflı sınıflandırma performans metriklerini hesaplayan sınıf."""

    @staticmethod
    def top_k_dogruluk(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        k_listesi: Tuple[int, ...] = (1, 2, 3),
    ) -> Dict[int, float]:
        """Her bir k değeri için Top-k doğruluğunu hesaplar.

        Args:
            y_true: (N,) int etiket dizisi.
            y_probs: (N, C) olasılık matrisi.
            k_listesi: Hesaplanacak k değerleri (örn. 1, 2, 3).

        Returns:
            Dict[int, float]: {1: top1_acc, 2: top2_acc, ...}
        """
        N, C = y_probs.shape
        sonuclar = {}

        # Olasılıkları azalan sırada sırala
        top_k_indices = np.argsort(y_probs, axis=1)[:, ::-1]

        for k in k_listesi:
            if k > C:
                continue
            # İlk k tahmin içinde gerçek etiket var mı?
            k_tahminler = top_k_indices[:, :k]
            dogru_sayisi = sum(y_true[i] in k_tahminler[i] for i in range(N))
            sonuclar[k] = float(dogru_sayisi / N)

        return sonuclar

    @staticmethod
    def cok_sinifli_roc_egrileri(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        n_classes: int,
    ) -> Dict[str, Union[Dict[int, Tuple[np.ndarray, np.ndarray, float]], float]]:
        """Her sınıf için One-vs-Rest ROC eğrilerini ve Macro/Micro AUC değerlerini hesaplar."""
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if n_classes == 2 and y_bin.shape[1] == 1:
            y_bin = np.hstack([1 - y_bin, y_bin])

        sinif_roclari = {}
        for c in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, c], y_probs[:, c])
            roc_auc = auc(fpr, tpr)
            sinif_roclari[c] = (fpr, tpr, float(roc_auc))

        # Micro-average ROC
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_probs.ravel())
        auc_micro = float(auc(fpr_micro, tpr_micro))

        # Macro-average ROC AUC
        auc_macro = float(roc_auc_score(y_bin, y_probs, average="macro", multi_class="ovr"))

        return {
            "sinif_roclari": sinif_roclari,
            "micro": (fpr_micro, tpr_micro, auc_micro),
            "macro_auc": auc_macro,
        }

    @staticmethod
    def cok_sinifli_pr_egrileri(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        n_classes: int,
    ) -> Dict[str, Union[Dict[int, Tuple[np.ndarray, np.ndarray, float]], float]]:
        """Her sınıf için Precision-Recall (PR) eğrilerini ve Ortalama Hassasiyeti (AP) hesaplar."""
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if n_classes == 2 and y_bin.shape[1] == 1:
            y_bin = np.hstack([1 - y_bin, y_bin])

        sinif_prleri = {}
        for c in range(n_classes):
            prec, rec, _ = precision_recall_curve(y_bin[:, c], y_probs[:, c])
            ap = float(average_precision_score(y_bin[:, c], y_probs[:, c]))
            sinif_prleri[c] = (prec, rec, ap)

        # Micro-average PR
        prec_micro, rec_micro, _ = precision_recall_curve(y_bin.ravel(), y_probs.ravel())
        ap_micro = float(average_precision_score(y_bin, y_probs, average="micro"))
        ap_macro = float(average_precision_score(y_bin, y_probs, average="macro"))

        return {
            "sinif_prleri": sinif_prleri,
            "micro": (prec_micro, rec_micro, ap_micro),
            "macro_ap": ap_macro,
        }

    @classmethod
    def kapsamli_rapor(
        cls,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        sinif_isimleri: List[str],
    ) -> Dict:
        """Tüm sınıflandırma metriklerini tek bir yapılandırılmış raporda toplar."""
        y_pred = np.argmax(y_probs, axis=1)
        n_classes = len(sinif_isimleri)

        top_k = cls.top_k_dogruluk(y_true, y_probs, k_listesi=(1, 2, 3))
        roc_bilgi = cls.cok_sinifli_roc_egrileri(y_true, y_probs, n_classes)
        pr_bilgi = cls.cok_sinifli_pr_egrileri(y_true, y_probs, n_classes)

        acc = float(accuracy_score(y_true, y_pred))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_true, y_pred)

        # Sınıf bazında özet
        sinif_raporu = {}
        for c, ad in enumerate(sinif_isimleri):
            c_mask = y_true == c
            c_toplam = int(np.sum(c_mask))
            c_dogru = int(np.sum((y_pred == c) & c_mask))
            sinif_raporu[ad] = {
                "toplam": c_toplam,
                "dogru": c_dogru,
                "dogruluk": c_dogru / c_toplam if c_toplam > 0 else 0.0,
                "auc": roc_bilgi["sinif_roclari"][c][2],
                "ap": pr_bilgi["sinif_prleri"][c][2],
            }

        return {
            "dogruluk": acc,
            "f1_macro": f1_macro,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "top_k": top_k,
            "roc_bilgi": roc_bilgi,
            "pr_bilgi": pr_bilgi,
            "karisiklik_matrisi": cm,
            "sinif_raporu": sinif_raporu,
        }
