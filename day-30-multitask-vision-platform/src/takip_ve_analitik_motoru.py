"""
Çoklu Görev Takip ve Mekansal Analitik Motoru (Tracking & Spatial Analytics).
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


class TakipHedefi:
    """Mekansal Analitik Geçmişi Olan Takip Nesnesi."""

    def __init__(self, track_id: int, ilk_kutu: np.ndarray, ilk_embedding: np.ndarray, sinif_id: int):
        self.track_id = track_id
        self.kutu = ilk_kutu
        self.embedding = ilk_embedding
        self.sinif_id = sinif_id
        self.time_since_update = 0
        self.omur = 1

        # Merkez Koordinatı
        cx = (ilk_kutu[0] + ilk_kutu[2]) / 2.0
        cy = (ilk_kutu[1] + ilk_kutu[3]) / 2.0
        self.yörünge = [(float(cx), float(cy))]
        self.hiz_gecmisi = [0.0]

    def guncelle(self, yeni_kutu: np.ndarray, yeni_embedding: np.ndarray):
        self.kutu = yeni_kutu
        self.embedding = 0.8 * self.embedding + 0.2 * yeni_embedding
        self.embedding /= (np.linalg.norm(self.embedding) + 1e-6)

        cx = (yeni_kutu[0] + yeni_kutu[2]) / 2.0
        cy = (yeni_kutu[1] + yeni_kutu[3]) / 2.0
        prev_cx, prev_cy = self.yörünge[-1]
        hiz = float(np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2))

        self.yörünge.append((float(cx), float(cy)))
        self.hiz_gecmisi.append(hiz)
        if len(self.yörünge) > 40:
            self.yörünge.pop(0)
            self.hiz_gecmisi.pop(0)

        self.time_since_update = 0
        self.omur += 1


class CokluGorevTakipAnalitikMotoru:
    """
    Dedektör ve Re-ID başlıklarından gelen çıktıları birleştirip
    gerçek zamanlı nesne takibi, hız kestirimi ve mekansal analitik üretir.
    """

    def __init__(self, max_cosine_dist: float = 0.40, max_age: int = 15):
        self.max_cosine_dist = max_cosine_dist
        self.max_age = max_age
        self.takipciler: List[TakipHedefi] = []
        self._id_counter = 1

    def guncelle(
        self,
        tespit_kutulari: List[np.ndarray],
        tespit_embeddingleri: np.ndarray,
        tespit_siniflari: List[int]
    ) -> List[TakipHedefi]:
        """Kare kare online takip ve eşleme."""
        # 1. Takipçilerin yaşını artır
        for t in self.takipciler:
            t.time_since_update += 1

        if len(self.takipciler) == 0:
            for i, box in enumerate(tespit_kutulari):
                yeni_t = TakipHedefi(
                    track_id=self._id_counter,
                    ilk_kutu=box,
                    ilk_embedding=tespit_embeddingleri[i],
                    sinif_id=tespit_siniflari[i]
                )
                self._id_counter += 1
                self.takipciler.append(yeni_t)
            return self.takipciler

        if len(tespit_kutulari) == 0:
            self.takipciler = [t for t in self.takipciler if t.time_since_update <= self.max_age]
            return [t for t in self.takipciler if t.time_since_update == 0]

        # 2. Re-ID + IoU Birleşik Maliyet Matrisi
        n_t = len(self.takipciler)
        n_d = len(tespit_kutulari)
        maliyet = np.zeros((n_t, n_d), dtype=np.float32)

        for i, t in enumerate(self.takipciler):
            for j, d_box in enumerate(tespit_kutulari):
                # Kosinüs Mesafesi
                d_cos = 1.0 - float(np.dot(t.embedding, tespit_embeddingleri[j]))
                # IoU Mesafesi
                d_iou = 1.0 - self._iou(t.kutu, d_box)
                # Birleşik Maliyet
                maliyet[i, j] = 0.7 * d_cos + 0.3 * d_iou

        # 3. Macar Algoritması
        satir_ind, sutun_ind = linear_sum_assignment(maliyet)

        eslesen_t = set()
        eslesen_d = set()

        for r, c in zip(satir_ind, sutun_ind):
            if maliyet[r, c] < self.max_cosine_dist:
                self.takipciler[r].guncelle(tespit_kutulari[c], tespit_embeddingleri[c])
                eslesen_t.add(r)
                eslesen_d.add(c)

        # 4. Eşleşmeyen tespitlerden yeni takipçi oluştur
        for j in range(n_d):
            if j not in eslesen_d:
                yeni_t = TakipHedefi(
                    track_id=self._id_counter,
                    ilk_kutu=tespit_kutulari[j],
                    ilk_embedding=tespit_embeddingleri[j],
                    sinif_id=tespit_siniflari[j]
                )
                self._id_counter += 1
                self.takipciler.append(yeni_t)

        # 5. Zaman aşımına uğrayanları temizle
        self.takipciler = [t for t in self.takipciler if t.time_since_update <= self.max_age]

        return [t for t in self.takipciler if t.time_since_update == 0]

    @staticmethod
    def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        inter = w * h
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter
        return float(inter / max(union, 1e-6))
