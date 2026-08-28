"""Mean Average Precision (mAP@0.5 ve mAP@0.5:0.95) Metrik Motoru.

Bu modül; Nesne tespiti modellerinin standart COCO ve VOC değerlendirme metrikleri olan
mAP@0.5, mAP@0.75 ve 10 farklı IoU eşiğinin ortalaması olan mAP@0.5:0.95 skorlarını
bağımsız, matematiksel ve vektörize olarak hesaplar.
"""

from typing import Dict, List, Tuple
import numpy as np


class MAPHesaplayici:
    """Nesne tespiti modelleri için mAP ve AP metriklerini hesaplayan sınıf."""

    @staticmethod
    def iou_hesapla(box1: np.ndarray, box2: np.ndarray) -> float:
        """İki [x1, y1, x2, y2] kutusu arasındaki IoU değerini hesaplar."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        inter = w * h

        area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
        union = area1 + area2 - inter

        return float(inter / union) if union > 0 else 0.0

    @classmethod
    def sinif_ap_hesapla(
        cls,
        tahminler: List[Dict],
        gercekler: List[Dict],
        iou_esigi: float = 0.5,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Tek bir sınıf için belirtilen IoU eşiğinde Ortalama Hassasiyeti (AP) hesaplar."""
        if len(gercekler) == 0:
            return 0.0, np.array([]), np.array([])
        if len(tahminler) == 0:
            return 0.0, np.array([0.0]), np.array([0.0])

        # Tahminleri güven skoruna göre azalan sırala
        sirali_tahminler = sorted(tahminler, key=lambda x: x["score"], reverse=True)
        toplam_gt = len(gercekler)

        tp = np.zeros(len(sirali_tahminler))
        fp = np.zeros(len(sirali_tahminler))
        gt_eslesti = [False] * toplam_gt

        for t_idx, t in enumerate(sirali_tahminler):
            t_box = np.array(t["box"])
            en_iyi_iou = 0.0
            en_iyi_gt_idx = -1

            for g_idx, g in enumerate(gercekler):
                g_box = np.array(g["box"])
                iou = cls.iou_hesapla(t_box, g_box)
                if iou > en_iyi_iou:
                    en_iyi_iou = iou
                    en_iyi_gt_idx = g_idx

            if en_iyi_iou >= iou_esigi and en_iyi_gt_idx >= 0 and not gt_eslesti[en_iyi_gt_idx]:
                tp[t_idx] = 1.0
                gt_eslesti[en_iyi_gt_idx] = True
            else:
                fp[t_idx] = 1.0

        # Kümülatif TP ve FP
        tp_kumulatif = np.cumsum(tp)
        fp_kumulatif = np.cumsum(fp)

        recalls = tp_kumulatif / max(1, toplam_gt)
        precisions = tp_kumulatif / np.maximum(1e-7, (tp_kumulatif + fp_kumulatif))

        # Precision-Recall eğrisi altındaki alanı hesapla (11-noktalı veya tüm-noktalar İnterpolasyonu)
        recalls_padded = np.concatenate(([0.0], recalls, [1.0]))
        precisions_padded = np.concatenate(([1.0], precisions, [0.0]))

        # Monoton azalan hassasiyet zarfı (Precision Envelope)
        for i in range(len(precisions_padded) - 2, -1, -1):
            precisions_padded[i] = max(precisions_padded[i], precisions_padded[i + 1])

        # Değişim noktalarında integrasyon
        degisim_idx = np.where(recalls_padded[1:] != recalls_padded[:-1])[0]
        ap = float(np.sum((recalls_padded[degisim_idx + 1] - recalls_padded[degisim_idx]) * precisions_padded[degisim_idx + 1]))

        return ap, precisions, recalls

    @classmethod
    def kapsamli_map_hesapla(
        cls,
        tahmin_listesi: List[Dict],
        gercek_listesi: List[Dict],
        sinif_isimleri: List[str],
    ) -> Dict:
        """Tüm sınıflar için mAP@0.5, mAP@0.75 ve mAP@0.5:0.95 (COCO Metric) hesaplar."""
        iou_esikleri = np.linspace(0.50, 0.95, 10)  # 0.50, 0.55, ..., 0.95

        sinif_ap_05 = {}
        sinif_ap_075 = {}
        sinif_ap_coco = {}
        pr_egrileri = {}

        for c_id, c_ad in enumerate(sinif_isimleri):
            c_tahminler = [t for t in tahmin_listesi if t["class_id"] == c_id]
            c_gercekler = [g for g in gercek_listesi if g["class_id"] == c_id]

            ap_05, prec, rec = cls.sinif_ap_hesapla(c_tahminler, c_gercekler, iou_esigi=0.50)
            ap_075, _, _ = cls.sinif_ap_hesapla(c_tahminler, c_gercekler, iou_esigi=0.75)

            # 10 eşiğin ortalaması
            esik_apleri = []
            for th in iou_esikleri:
                ap_th, _, _ = cls.sinif_ap_hesapla(c_tahminler, c_gercekler, iou_esigi=float(th))
                esik_apleri.append(ap_th)

            ap_coco = float(np.mean(esik_apleri))

            sinif_ap_05[c_ad] = ap_05
            sinif_ap_075[c_ad] = ap_075
            sinif_ap_coco[c_ad] = ap_coco
            pr_egrileri[c_ad] = (prec, rec)

        map_05 = float(np.mean(list(sinif_ap_05.values())))
        map_075 = float(np.mean(list(sinif_ap_075.values())))
        map_coco = float(np.mean(list(sinif_ap_coco.values())))

        return {
            "map_05": map_05,
            "map_075": map_075,
            "map_05_95": map_coco,
            "sinif_ap_05": sinif_ap_05,
            "sinif_ap_075": sinif_ap_075,
            "sinif_ap_coco": sinif_ap_coco,
            "pr_egrileri": pr_egrileri,
        }
