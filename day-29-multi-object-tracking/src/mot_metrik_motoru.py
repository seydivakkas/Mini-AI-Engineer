"""
Çoklu Nesne Takibi Değerlendirme Metrikleri:
MOTA (Multiple Object Tracking Accuracy), IDF1 (ID F1-Score), IDSW (Kimlik Değişimi), MT/ML.
"""

from typing import Dict, List, Tuple, Any
import numpy as np


class MOTMetrikMotoru:
    """CLEAR MOT ve ID Metrikleri Hesaplama Motoru."""

    @classmethod
    def iou(cls, box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        kesisim = w * h
        alan1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        alan2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        birlesim = alan1 + alan2 - kesisim
        if birlesim <= 0:
            return 0.0
        return float(kesisim / birlesim)

    @classmethod
    def degerlendir_video(
        cls,
        kareler_gt: List[List[Dict[str, Any]]],
        kareler_tahmin: List[List[Dict[str, Any]]],
        iou_esigi: float = 0.5
    ) -> Dict[str, Any]:
        """
        kareler_gt: Her kare için list of dict: [{'id': 1, 'box': [x1,y1,x2,y2]}, ...]
        kareler_tahmin: Her kare için list of dict: [{'id': 101, 'box': [x1,y1,x2,y2]}, ...]
        """
        toplam_gt = 0
        toplam_fp = 0
        toplam_fn = 0
        toplam_idsw = 0

        # GT ID -> Son Eşleşen Pred ID Haritası
        gt_to_pred_haritasi = {}
        # Global ID TP/FP/FN hesapları (IDF1 için)
        id_tp = 0
        id_fp = 0
        id_fn = 0

        gt_izleme_sureleri = {}  # gt_id -> {'toplam_kare': int, 'izlenen_kare': int}

        for f_idx, (gts, preds) in enumerate(zip(kareler_gt, kareler_tahmin)):
            toplam_gt += len(gts)

            # Karedeki GT'lerin ömür takibi
            for g in gts:
                gid = g["id"]
                if gid not in gt_izleme_sureleri:
                    gt_izleme_sureleri[gid] = {"toplam": 0, "izlenen": 0}
                gt_izleme_sureleri[gid]["toplam"] += 1

            if len(gts) == 0:
                toplam_fp += len(preds)
                id_fp += len(preds)
                continue

            if len(preds) == 0:
                toplam_fn += len(gts)
                id_fn += len(gts)
                continue

            # Eşleşme Matrisi
            iou_matrisi = np.zeros((len(gts), len(preds)))
            for i, g in enumerate(gts):
                for j, p in enumerate(preds):
                    iou_matrisi[i, j] = cls.iou(np.array(g["box"]), np.array(p["box"]))

            eslesen_gts = set()
            eslesen_preds = set()

            # Açgözlü en yüksek IoU eşlemesi
            while True:
                max_val = np.max(iou_matrisi)
                if max_val < iou_esigi:
                    break
                r, c = np.unravel_index(np.argmax(iou_matrisi), iou_matrisi.shape)
                if r in eslesen_gts or c in eslesen_preds:
                    iou_matrisi[r, c] = 0.0
                    continue

                gt_id = gts[r]["id"]
                pred_id = preds[c]["id"]

                # ID Switch Kontrolü
                if gt_id in gt_to_pred_haritasi and gt_to_pred_haritasi[gt_id] != pred_id:
                    toplam_idsw += 1

                gt_to_pred_haritasi[gt_id] = pred_id
                gt_izleme_sureleri[gt_id]["izlenen"] += 1

                eslesen_gts.add(r)
                eslesen_preds.add(c)
                id_tp += 1
                iou_matrisi[r, :] = 0.0
                iou_matrisi[:, c] = 0.0

            fn = len(gts) - len(eslesen_gts)
            fp = len(preds) - len(eslesen_preds)

            toplam_fn += fn
            toplam_fp += fp
            id_fn += fn
            id_fp += fp

        # Metrik Hesaplamaları
        mota = 1.0 - (toplam_fn + toplam_fp + toplam_idsw) / max(toplam_gt, 1)
        den = (2 * id_tp + id_fp + id_fn)
        idf1 = (2 * id_tp / den) if den > 0 else 0.0
        precision = (id_tp / (id_tp + id_fp)) if (id_tp + id_fp) > 0 else 0.0
        recall = (id_tp / (id_tp + id_fn)) if (id_tp + id_fn) > 0 else 0.0

        # Mostly Tracked (MT >= %80) ve Mostly Lost (ML < %20)
        mt = 0
        ml = 0
        for gid, stats in gt_izleme_sureleri.items():
            oran = stats["izlenen"] / max(stats["toplam"], 1)
            if oran >= 0.8:
                mt += 1
            elif oran < 0.2:
                ml += 1

        return {
            "MOTA": float(mota),
            "IDF1": float(idf1),
            "IDSW": int(toplam_idsw),
            "FP": int(toplam_fp),
            "FN": int(toplam_fn),
            "Toplam_GT": int(toplam_gt),
            "Hassasiyet": float(precision),
            "Anma": float(recall),
            "MT": int(mt),
            "ML": int(ml),
            "Toplam_Hedef": len(gt_izleme_sureleri)
        }
