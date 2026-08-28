"""Bounding Box Format Dönüştürücü ve Geometrik Yardımcı Modülü.

Bu modül; Pascal VOC (xyxy), COCO (xywh) ve YOLO (cxcywh) koordinat formatları arasında
vektörize çift yönlü dönüşümleri, normalizasyon ve görsel sınır kırpma işlemlerini sağlar.
"""

from typing import Union
import numpy as np
import torch


class KutuDonusturucu:
    """Bounding Box koordinat dönüşümlerini gerçekleştiren sınıf."""

    @staticmethod
    def xyxy_to_xywh(
        boxes: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """[x1, y1, x2, y2] -> [x1, y1, width, height] formatına dönüştürür."""
        if isinstance(boxes, torch.Tensor):
            x1, y1, x2, y2 = boxes.unbind(-1)
            w = x2 - x1
            h = y2 - y1
            return torch.stack([x1, y1, w, h], dim=-1)
        else:
            x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
            w = x2 - x1
            h = y2 - y1
            return np.stack([x1, y1, w, h], axis=-1)

    @staticmethod
    def xywh_to_xyxy(
        boxes: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """[x1, y1, width, height] -> [x1, y1, x2, y2] formatına dönüştürür."""
        if isinstance(boxes, torch.Tensor):
            x1, y1, w, h = boxes.unbind(-1)
            x2 = x1 + w
            y2 = y1 + h
            return torch.stack([x1, y1, x2, y2], dim=-1)
        else:
            x1, y1, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
            x2 = x1 + w
            y2 = y1 + h
            return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def xyxy_to_cxcywh(
        boxes: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """[x1, y1, x2, y2] -> [cx, cy, width, height] formatına dönüştürür."""
        if isinstance(boxes, torch.Tensor):
            x1, y1, x2, y2 = boxes.unbind(-1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            return torch.stack([cx, cy, w, h], dim=-1)
        else:
            x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            return np.stack([cx, cy, w, h], axis=-1)

    @staticmethod
    def cxcywh_to_xyxy(
        boxes: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """[cx, cy, width, height] -> [x1, y1, x2, y2] formatına dönüştürür."""
        if isinstance(boxes, torch.Tensor):
            cx, cy, w, h = boxes.unbind(-1)
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            return torch.stack([x1, y1, x2, y2], dim=-1)
        else:
            cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def normalize_et(
        boxes: np.ndarray, img_w: int, img_h: int
    ) -> np.ndarray:
        """Piksel koordinatlarını [0, 1] aralığına normalize eder."""
        norm_boxes = boxes.copy().astype(float)
        norm_boxes[..., [0, 2]] /= img_w
        norm_boxes[..., [1, 3]] /= img_h
        return norm_boxes

    @staticmethod
    def denormalize_et(
        boxes: np.ndarray, img_w: int, img_h: int
    ) -> np.ndarray:
        """Normalize [0, 1] koordinatlarını piksel değerlerine dönüştürür."""
        denorm = boxes.copy().astype(float)
        denorm[..., [0, 2]] *= img_w
        denorm[..., [1, 3]] *= img_h
        return denorm

    @staticmethod
    def kirp_sinirla(
        boxes: np.ndarray, img_w: int, img_h: int
    ) -> np.ndarray:
        """Kutuları görsel sınırları [0, img_w], [0, img_h] içine kırpar."""
        clipped = boxes.copy()
        clipped[..., [0, 2]] = np.clip(clipped[..., [0, 2]], 0, img_w)
        clipped[..., [1, 3]] = np.clip(clipped[..., [1, 3]], 0, img_h)
        return clipped
