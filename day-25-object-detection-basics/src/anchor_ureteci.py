"""Anchor Box Üretimi, IoU Eşleme ve Bounding Box Regresyon Modülü.

Bu modül; Çok ölçekli feature map gridleri üzerinde anchor kutuları üretir,
Ground Truth kutuları ile IoU eşlemesi (Anchor-to-GT Matching) yapar ve
YOLO / SSD / Faster R-CNN regresyon parametrelerini (Delta t_x, t_y, t_w, t_h) hesaplar.
"""

from typing import Dict, List, Tuple
import numpy as np
from src.kutu_donusturucu import KutuDonusturucu
from src.iou_hesaplayici import IoUHesaplayici


class AnchorUreteci:
    """Anchor kutuları üreten ve regresyon hedeflerini hesaplayan sınıf."""

    @staticmethod
    def anchor_grid_uret(
        grid_w: int,
        grid_h: int,
        stride: int,
        olcekler: Tuple[float, ...] = (32.0, 64.0, 128.0),
        en_boy_oranlari: Tuple[float, ...] = (0.5, 1.0, 2.0),
    ) -> np.ndarray:
        """Belirtilen feature map ızgarası üzerinde tüm anchor kutularını [x1, y1, x2, y2] formatında üretir."""
        anchors = []

        # Her hücre merkezinin piksel koordinatları
        shift_x = (np.arange(0, grid_w) + 0.5) * stride
        shift_y = (np.arange(0, grid_h) + 0.5) * stride
        grid_x, grid_y = np.meshgrid(shift_x, shift_y)
        grid_x = grid_x.ravel()
        grid_y = grid_y.ravel()

        for s in olcekler:
            for r in en_boy_oranlari:
                # w = s * sqrt(r), h = s / sqrt(r)
                w = s * np.sqrt(r)
                h = s / np.sqrt(r)

                x1 = grid_x - w / 2.0
                y1 = grid_y - h / 2.0
                x2 = grid_x + w / 2.0
                y2 = grid_y + h / 2.0

                kutu_seti = np.stack([x1, y1, x2, y2], axis=-1)
                anchors.append(kutu_seti)

        # (K, grid_h * grid_w, 4) -> (Toplam_Anchor_Sayisi, 4)
        tum_anchors = np.concatenate(anchors, axis=0)
        return tum_anchors

    @staticmethod
    def ground_truth_esle(
        anchors: np.ndarray,
        gt_boxes: np.ndarray,
        pos_iou_esigi: float = 0.5,
        neg_iou_esigi: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Anchor kutuları ile Ground Truth kutularını eşler."""
        num_anchors = len(anchors)
        num_gt = len(gt_boxes)

        if num_gt == 0:
            etiketler = np.zeros(num_anchors, dtype=int)
            eslesen_gt_indeksleri = np.full(num_anchors, -1, dtype=int)
            max_ious = np.zeros(num_anchors, dtype=float)
            return etiketler, eslesen_gt_indeksleri, max_ious

        iou_mat = IoUHesaplayici.iou_matrisi(anchors, gt_boxes)  # (N, M)

        # Her anchor için en yüksek IoU'ya sahip GT
        max_ious = np.max(iou_mat, axis=1)
        eslesen_gt_indeksleri = np.argmax(iou_mat, axis=1)

        # Başlangıçta nötr (-1)
        etiketler = np.full(num_anchors, -1, dtype=int)

        # Negatif (Arka plan) = 0
        etiketler[max_ious < neg_iou_esigi] = 0

        # Pozitif (Ön plan) = 1
        etiketler[max_ious >= pos_iou_esigi] = 1

        # Kural 2: Her GT için en yüksek IoU'ya sahip anchor MUTLAKA pozitif olmalıdır
        gt_en_iyi_anchor_indeksleri = np.argmax(iou_mat, axis=0)
        etiketler[gt_en_iyi_anchor_indeksleri] = 1

        return etiketler, eslesen_gt_indeksleri, max_ious

    @staticmethod
    def kutu_regresyon_hedefleri(
        anchors: np.ndarray, gt_boxes: np.ndarray
    ) -> np.ndarray:
        """Anchor kutuları ile GT kutuları arasındaki regresyon delta hedeflerini hesaplar."""
        # xyxy -> cxcywh
        a_cxcywh = KutuDonusturucu.xyxy_to_cxcywh(anchors)
        g_cxcywh = KutuDonusturucu.xyxy_to_cxcywh(gt_boxes)

        tx = (g_cxcywh[:, 0] - a_cxcywh[:, 0]) / np.maximum(a_cxcywh[:, 2], 1e-7)
        ty = (g_cxcywh[:, 1] - a_cxcywh[:, 1]) / np.maximum(a_cxcywh[:, 3], 1e-7)
        tw = np.log(np.maximum(g_cxcywh[:, 2], 1e-7) / np.maximum(a_cxcywh[:, 2], 1e-7))
        th = np.log(np.maximum(g_cxcywh[:, 3], 1e-7) / np.maximum(a_cxcywh[:, 3], 1e-7))

        deltalar = np.stack([tx, ty, tw, th], axis=-1)
        return deltalar

    @staticmethod
    def regresyondan_kutulari_coz(
        anchors: np.ndarray, deltalar: np.ndarray
    ) -> np.ndarray:
        """Tahmin edilen regresyon deltalarını anchor kutuları üzerine uygulayarak [x1, y1, x2, y2] üretir."""
        a_cxcywh = KutuDonusturucu.xyxy_to_cxcywh(anchors)

        pred_cx = deltalar[:, 0] * a_cxcywh[:, 2] + a_cxcywh[:, 0]
        pred_cy = deltalar[:, 1] * a_cxcywh[:, 3] + a_cxcywh[:, 1]
        pred_w = a_cxcywh[:, 2] * np.exp(np.clip(deltalar[:, 2], -10.0, 10.0))
        pred_h = a_cxcywh[:, 3] * np.exp(np.clip(deltalar[:, 3], -10.0, 10.0))

        pred_cxcywh = np.stack([pred_cx, pred_cy, pred_w, pred_h], axis=-1)
        pred_xyxy = KutuDonusturucu.cxcywh_to_xyxy(pred_cxcywh)
        return pred_xyxy
