"""
İleri Düzey Bölütleme Metrikleri:
Maske IoU, Örnek Tabanlı AP (Instance AP_50, AP_75) ve Panoptik Kalite (Panoptic Quality - PQ = SQ x RQ).
"""

from typing import Dict, List, Tuple, Any
import numpy as np


class MaskeIoUHesaplayici:
    """Tekil ikili maskeler arasında vektörize IoU hesaplayıcı."""

    @staticmethod
    def iou(mask1: np.ndarray, mask2: np.ndarray, eps: float = 1e-6) -> float:
        m1 = (mask1 > 0.5).astype(bool)
        m2 = (mask2 > 0.5).astype(bool)
        kesisim = np.logical_and(m1, m2).sum()
        birlesim = np.logical_or(m1, m2).sum()
        if birlesim == 0:
            return 1.0 if kesisim == 0 else 0.0
        return float((kesisim + eps) / (birlesim + eps))

    @staticmethod
    def pairwise_mask_iou(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> np.ndarray:
        n_pred = len(pred_masks)
        n_gt = len(gt_masks)
        iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
        for i in range(n_pred):
            for j in range(n_gt):
                iou_matrix[i, j] = MaskeIoUHesaplayici.iou(pred_masks[i], gt_masks[j])
        return iou_matrix


class PanoptikMetrikHesaplayici:
    """
    Panoptik Kalite (Panoptic Quality - PQ) Hesaplayıcı (Kirillov et al., 2019):
    PQ = SQ (Segmentation Quality) * RQ (Recognition Quality)
    """

    @classmethod
    def hesapla_pq(
        cls,
        pred_panoptic: np.ndarray,
        gt_panoptic: np.ndarray,
        iou_esigi: float = 0.5
    ) -> Dict[str, Any]:
        """
        Tahmin ve Gerçek panoptik haritaları üzerinden sınıf bazında ve genel PQ, SQ, RQ hesaplar.
        """
        pred_ids = np.unique(pred_panoptic)
        gt_ids = np.unique(gt_panoptic)

        # 0 (arka plan / tanımsız) harici sınıfları grupla
        # panoptic_id // 1000 = kategori_id
        tum_kategoriler = set([pid // 1000 for pid in gt_ids if pid > 0] + [pid // 1000 for pid in pred_ids if pid > 0])

        kategori_metrikleri = {}
        toplam_pq_list = []
        toplam_sq_list = []
        toplam_rq_list = []

        for cat_id in tum_kategoriler:
            cat_gt_ids = [pid for pid in gt_ids if pid // 1000 == cat_id]
            cat_pred_ids = [pid for pid in pred_ids if pid // 1000 == cat_id]

            tp = 0
            fp = 0
            fn = 0
            iou_toplami = 0.0

            eslesmis_gt = set()
            eslesmis_pred = set()

            # Eşleşmeleri bul (IoU > 0.5 garantisi tekil eşleşmeyi sağlar)
            for p_id in cat_pred_ids:
                p_mask = (pred_panoptic == p_id)
                for g_id in cat_gt_ids:
                    if g_id in eslesmis_gt:
                        continue
                    g_mask = (gt_panoptic == g_id)
                    iou_val = MaskeIoUHesaplayici.iou(p_mask, g_mask)
                    if iou_val > iou_esigi:
                        tp += 1
                        iou_toplami += iou_val
                        eslesmis_gt.add(g_id)
                        eslesmis_pred.add(p_id)
                        break

            fp = len(cat_pred_ids) - len(eslesmis_pred)
            fn = len(cat_gt_ids) - len(eslesmis_gt)

            # SQ ve RQ hesapla
            sq = (iou_toplami / tp) if tp > 0 else 0.0
            rq = (tp / (tp + 0.5 * fp + 0.5 * fn)) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0
            pq = sq * rq

            kategori_metrikleri[cat_id] = {
                "PQ": float(pq),
                "SQ": float(sq),
                "RQ": float(rq),
                "TP": tp,
                "FP": fp,
                "FN": fn
            }
            toplam_pq_list.append(pq)
            toplam_sq_list.append(sq)
            toplam_rq_list.append(rq)

        genel_pq = float(np.mean(toplam_pq_list)) if toplam_pq_list else 0.0
        genel_sq = float(np.mean(toplam_sq_list)) if toplam_sq_list else 0.0
        genel_rq = float(np.mean(toplam_rq_list)) if toplam_rq_list else 0.0

        return {
            "genel_pq": genel_pq,
            "genel_sq": genel_sq,
            "genel_rq": genel_rq,
            "kategori_bazinda": kategori_metrikleri
        }

    @classmethod
    def hesapla_instance_ap(
        cls,
        pred_masks: List[np.ndarray],
        pred_scores: List[float],
        pred_labels: List[int],
        gt_masks: List[np.ndarray],
        gt_labels: List[int],
        iou_esikleri: List[float] = [0.50, 0.75]
    ) -> Dict[str, float]:
        """
        Örnek tabanlı maske tespitinde AP_50 ve AP_75 hesaplar.
        """
        if not gt_masks:
            return {"AP_50": 0.0, "AP_75": 0.0, "mAP_mask": 0.0}

        sonuclar = {}
        for th in iou_esikleri:
            eslesme = 0
            kullanilan_gt = set()
            # Skorlara göre sırala
            sirali_idx = np.argsort(pred_scores)[::-1]
            for p_idx in sirali_idx:
                p_mask = pred_masks[p_idx]
                p_lbl = pred_labels[p_idx]
                en_iyi_iou = 0.0
                en_iyi_gt_idx = -1
                for g_idx, (g_mask, g_lbl) in enumerate(zip(gt_masks, gt_labels)):
                    if g_idx in kullanilan_gt or p_lbl != g_lbl:
                        continue
                    cur_iou = MaskeIoUHesaplayici.iou(p_mask, g_mask)
                    if cur_iou > en_iyi_iou:
                        en_iyi_iou = cur_iou
                        en_iyi_gt_idx = g_idx
                if en_iyi_iou >= th and en_iyi_gt_idx != -1:
                    eslesme += 1
                    kullanilan_gt.add(en_iyi_gt_idx)

            precision = eslesme / max(len(pred_masks), 1)
            recall = eslesme / max(len(gt_masks), 1)
            f1 = (2 * precision * recall) / (precision + recall + 1e-6)
            th_key = f"AP_{int(th*100)}"
            sonuclar[th_key] = float(precision)

        sonuclar["mAP_mask"] = float(np.mean(list(sonuclar.values())))
        return sonuclar
