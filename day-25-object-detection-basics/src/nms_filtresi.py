"""Non-Maximum Suppression (NMS) ve Soft-NMS Filtreleme Modülü.

Bu modül; nesne tespit modellerinin ürettiği mükerrer/çakışan aday kutuları
(Bounding Box Proposals) güven skorlarına ve IoU eşiklerine göre eleyerek
en doğru tespitleri izole eden Klasik NMS, Sınıfa Duyarlı NMS ve Soft-NMS algoritmalarını içerir.
"""

from typing import List, Tuple, Union
import numpy as np
from src.iou_hesaplayici import IoUHesaplayici


class NMSFiltresi:
    """Aday kutularda fazlalıkları temizleyen Non-Maximum Suppression filtresi."""

    @staticmethod
    def klasik_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_esigi: float = 0.5,
        skor_esigi: float = 0.25,
    ) -> List[int]:
        """Açgözlü (Greedy) klasik NMS algoritması."""
        if len(boxes) == 0:
            return []

        # Skor eşiğinin altındakileri ele
        gecerli_mask = scores >= skor_esigi
        indeksler = np.where(gecerli_mask)[0]

        if len(indeksler) == 0:
            return []

        gecerli_boxes = boxes[indeksler]
        gecerli_scores = scores[indeksler]

        # Skorlara göre azalan sırala
        sirali_sira = np.argsort(gecerli_scores)[::-1]
        secilenler = []

        while len(sirali_sira) > 0:
            en_iyi_idx = sirali_sira[0]
            secilenler.append(indeksler[en_iyi_idx])

            if len(sirali_sira) == 1:
                break

            kalan_siralar = sirali_sira[1:]
            en_iyi_kutu = gecerli_boxes[en_iyi_idx : en_iyi_idx + 1]
            kalan_kutular = gecerli_boxes[kalan_siralar]

            iou = IoUHesaplayici.iou_matrisi(en_iyi_kutu, kalan_kutular).reshape(-1)

            # IoU eşiğinin altındaki kutuları koru
            tutulanlar = np.where(iou <= iou_esigi)[0]
            sirali_sira = kalan_siralar[tutulanlar]

        return secilenler

    @classmethod
    def sinifa_duyarli_nms(
        cls,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        iou_esigi: float = 0.5,
        skor_esigi: float = 0.25,
    ) -> List[int]:
        """Farklı sınıflara ait kutuların birbirini ezmesini engelleyen sınıfa duyarlı NMS."""
        if len(boxes) == 0:
            return []

        benzersiz_siniflar = np.unique(labels)
        tum_secilenler = []

        for c in benzersiz_siniflar:
            sinif_mask = labels == c
            sinif_indeksleri = np.where(sinif_mask)[0]

            if len(sinif_indeksleri) == 0:
                continue

            secilen_yerel = cls.klasik_nms(
                boxes[sinif_indeksleri],
                scores[sinif_indeksleri],
                iou_esigi=iou_esigi,
                skor_esigi=skor_esigi,
            )
            tum_secilenler.extend(sinif_indeksleri[secilen_yerel])

        return sorted(tum_secilenler)

    @staticmethod
    def soft_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_esigi: float = 0.5,
        sigma: float = 0.5,
        skor_esigi: float = 0.05,
        method: str = "gaussian",
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Soft-NMS algoritması: Çakışan kutuları tamamen silmek yerine güven skorlarını düşürür."""
        N = len(boxes)
        if N == 0:
            return np.zeros((0, 4)), np.zeros(0), []

        b_boxes = boxes.copy().astype(float)
        b_scores = scores.copy().astype(float)
        orijinal_indeksler = list(range(N))

        secilen_indeksler = []

        for i in range(N):
            # En yüksek skorlu kutuyu bul
            max_pos = i + np.argmax(b_scores[i:])
            # Takas et
            b_boxes[[i, max_pos]] = b_boxes[[max_pos, i]]
            b_scores[[i, max_pos]] = b_scores[[max_pos, i]]
            orijinal_indeksler[i], orijinal_indeksler[max_pos] = orijinal_indeksler[max_pos], orijinal_indeksler[i]

            en_iyi_kutu = b_boxes[i : i + 1]
            kalan_kutular = b_boxes[i + 1 :]

            if len(kalan_kutular) > 0:
                ious = IoUHesaplayici.iou_matrisi(en_iyi_kutu, kalan_kutular).reshape(-1)

                if method == "gaussian":
                    decay = np.exp(-(ious ** 2) / sigma)
                    b_scores[i + 1 :] *= decay
                else:  # Linear decay
                    decay = np.where(ious > iou_esigi, 1.0 - ious, 1.0)
                    b_scores[i + 1 :] *= decay

        # Skor eşiğinin üzerindeki indeksleri seç
        gecerli = np.where(b_scores >= skor_esigi)[0]
        final_indeksler = [orijinal_indeksler[idx] for idx in gecerli]

        return b_boxes[gecerli], b_scores[gecerli], final_indeksler
