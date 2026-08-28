"""Intersection over Union (IoU), GIoU, DIoU ve CIoU Hesaplayıcı Modülü.

Bu modül; nesne tespitinde iki kutu kümesi arasındaki çakışma oranını,
genelleştirilmiş IoU (GIoU), mesafe tabanlı IoU (DIoU) ve tam IoU (CIoU)
metriklerini vektörize biçimde hesaplar.
"""

from typing import Tuple, Union
import numpy as np
import torch


class IoUHesaplayici:
    """Bounding Box çakışma ve mesafe metriklerini hesaplayan sınıf."""

    @staticmethod
    def kutu_alani(boxes: np.ndarray) -> np.ndarray:
        """[x1, y1, x2, y2] formatındaki kutuların alanlarını hesaplar."""
        w = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        h = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        return w * h

    @classmethod
    def iou_matrisi(
        cls, boxes1: np.ndarray, boxes2: np.ndarray
    ) -> np.ndarray:
        """(N, 4) ve (M, 4) boyutundaki kutu dizileri arasındaki pairwise IoU matrisini (N, M) hesaplar."""
        N = boxes1.shape[0]
        M = boxes2.shape[0]

        if N == 0 or M == 0:
            return np.zeros((N, M), dtype=float)

        # Kesişim alanının sol-üst ve sağ-alt koordinatları
        # boxes1[:, None, :2] -> (N, 1, 2)
        # boxes2[None, :, :2] -> (1, M, 2)
        lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
        rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)

        wh = np.maximum(0.0, rb - lt)  # (N, M, 2)
        inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M) kesişim alanı

        alan1 = cls.kutu_alani(boxes1)[:, None]  # (N, 1)
        alan2 = cls.kutu_alani(boxes2)[None, :]  # (1, M)

        union = alan1 + alan2 - inter  # (N, M) birleşim alanı
        iou = np.where(union > 0, inter / np.maximum(union, 1e-7), 0.0)
        return iou

    @classmethod
    def giou_matrisi(
        cls, boxes1: np.ndarray, boxes2: np.ndarray
    ) -> np.ndarray:
        """Generalized IoU (GIoU) matrisini (N, M) hesaplar (Ayrık kutular için de gradyan sağlar)."""
        N = boxes1.shape[0]
        M = boxes2.shape[0]

        if N == 0 or M == 0:
            return np.zeros((N, M), dtype=float)

        # Kesişim ve Birleşim
        lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = np.maximum(0.0, rb - lt)
        inter = wh[:, :, 0] * wh[:, :, 1]

        alan1 = cls.kutu_alani(boxes1)[:, None]
        alan2 = cls.kutu_alani(boxes2)[None, :]
        union = alan1 + alan2 - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-7), 0.0)

        # Her iki kutuyu da içine alan en küçük çevreleyen kutu (Enclosing Box C)
        c_lt = np.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
        c_rb = np.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        c_wh = np.maximum(0.0, c_rb - c_lt)
        c_area = c_wh[:, :, 0] * c_wh[:, :, 1]

        giou = iou - ((c_area - union) / np.maximum(c_area, 1e-7))
        return giou

    @classmethod
    def diou_matrisi(
        cls, boxes1: np.ndarray, boxes2: np.ndarray
    ) -> np.ndarray:
        """Distance-IoU (DIoU) matrisini (N, M) hesaplar (Merkez noktaları arası Öklid mesafesi cezası)."""
        N = boxes1.shape[0]
        M = boxes2.shape[0]

        if N == 0 or M == 0:
            return np.zeros((N, M), dtype=float)

        iou = cls.iou_matrisi(boxes1, boxes2)

        # Merkez koordinatları
        b1_cx = (boxes1[:, 0] + boxes1[:, 2]) / 2.0
        b1_cy = (boxes1[:, 1] + boxes1[:, 3]) / 2.0
        b2_cx = (boxes2[:, 0] + boxes2[:, 2]) / 2.0
        b2_cy = (boxes2[:, 1] + boxes2[:, 3]) / 2.0

        # Merkezler arası mesafe karesi rho^2
        rho2 = (b1_cx[:, None] - b2_cx[None, :]) ** 2 + (b1_cy[:, None] - b2_cy[None, :]) ** 2

        # En küçük çevreleyen kutunun köşegen uzunluğu karesi c^2
        c_lt = np.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
        c_rb = np.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        c_wh = np.maximum(0.0, c_rb - c_lt)
        c2 = c_wh[:, :, 0] ** 2 + c_wh[:, :, 1] ** 2

        diou = iou - (rho2 / np.maximum(c2, 1e-7))
        return diou
