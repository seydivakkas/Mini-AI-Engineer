"""
DeepSORT Takipçi Yaşam Döngüsü ve Macar Algoritması Eşleme Yöneticisi.
"""

from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman_filtresi import KalmanKutuFiltresi
from .reid_cikarici import ReIDEmbeddingCikarici


class TakipDurumu(Enum):
    TENTATIVE = 1  # Deneme Aşaması (n_init tespit gerektirir)
    CONFIRMED = 2  # Onaylı Aktif Takipçi
    DELETED = 3    # Silinmiş / Kayıp Takipçi


class Takipci:
    """Tekil bir nesnenin Kalman durumu, Re-ID galerisi ve yörünge geçmişi."""

    _id_sayaci = 1

    def __init__(
        self,
        ilk_kutu: np.ndarray,
        ilk_embedding: np.ndarray,
        kalman_filtresi: KalmanKutuFiltresi,
        n_init: int = 3,
        max_age: int = 30
    ):
        self.track_id = Takipci._id_sayaci
        Takipci._id_sayaci += 1

        self.kalman = kalman_filtresi
        self.mean, self.covariance = self.kalman.ilklendir(ilk_kutu)

        self.state = TakipDurumu.TENTATIVE
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.n_init = n_init
        self.max_age = max_age

        # Re-ID Galeri Havuzu (Son 100 embedding)
        self.features: List[np.ndarray] = [ilk_embedding]

        # Yörünge Çizimi için Merkez Noktaları
        u, v, _, _ = self.mean[:4]
        self.yörünge: List[Tuple[float, float]] = [(float(u), float(v))]

    def tahmin_et(self):
        """Kalman tahmin adımı."""
        self.mean, self.covariance = self.kalman.tahmin(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def guncelle(self, olcum: np.ndarray, embedding: np.ndarray):
        """Kalman güncelleme adımı ve durum geçişi."""
        self.mean, self.covariance = self.kalman.guncelle(self.mean, self.covariance, olcum)
        self.features.append(embedding)
        if len(self.features) > 100:
            self.features.pop(0)

        self.hits += 1
        self.time_since_update = 0

        if self.state == TakipDurumu.TENTATIVE and self.hits >= self.n_init:
            self.state = TakipDurumu.CONFIRMED

        u, v, _, _ = self.mean[:4]
        self.yörünge.append((float(u), float(v)))
        if len(self.yörünge) > 50:
            self.yörünge.pop(0)

    def silindi_isaretle(self):
        self.state = TakipDurumu.DELETED

    def guncel_kutu(self) -> np.ndarray:
        """Mevcut [x1, y1, x2, y2] kutu koordinatları."""
        return self.kalman.uvgh_to_kutu(self.mean[:4])


class DeepSORTTakipci:
    """
    DeepSORT Çoklu Nesne Takipçisi:
    Kalman Durum Tahmini + Re-ID Görsel Görünüş + Mahalanobis Kapılama + Macar Algoritması.
    """

    def __init__(
        self,
        max_cosine_distance: float = 0.35,
        nn_budget: int = 100,
        max_age: int = 30,
        n_init: int = 3
    ):
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.max_age = max_age
        self.n_init = n_init

        self.kalman = KalmanKutuFiltresi()
        self.takipciler: List[Takipci] = []
        Takipci._id_sayaci = 1

    def adim(self, tespitler: List[np.ndarray], embeddings: np.ndarray) -> List[Takipci]:
        """
        Her video karesi için tam takip döngüsü:
        1. Tüm takipçileri ileri besle (Predict)
        2. Eşleme Maliyet Matrisini oluştur (Cosine + Mahalanobis Gating)
        3. Macar Algoritması ile eşle
        4. Eşleşenleri güncelle, eşleşmeyenleri sil/yeni başlat
        """
        # 1. Tahmin
        for t in self.takipciler:
            t.tahmin_et()

        # 2. Eşleme (Matching)
        eslesmeler, eslesmeyen_takipciler, eslesmeyen_tespitler = self._esle(tespitler, embeddings)

        # 3. Eşleşen Takipçileri Güncelle
        for t_idx, d_idx in eslesmeler:
            self.takipciler[t_idx].guncelle(tespitler[d_idx], embeddings[d_idx])

        # 4. Eşleşmeyen Takipçilerin Kontrolü
        for t_idx in eslesmeyen_takipciler:
            t = self.takipciler[t_idx]
            if t.time_since_update > self.max_age:
                t.silindi_isaretle()

        # 5. Eşleşmeyen Tespitlerden Yeni Takipçi Başlat
        for d_idx in eslesmeyen_tespitler:
            yeni_t = Takipci(
                ilk_kutu=tespitler[d_idx],
                ilk_embedding=embeddings[d_idx],
                kalman_filtresi=self.kalman,
                n_init=self.n_init,
                max_age=self.max_age
            )
            self.takipciler.append(yeni_t)

        # 6. Silinenleri Temizle
        self.takipciler = [t for t in self.takipciler if t.state != TakipDurumu.DELETED]

        # Yalnızca onaylı aktif takipçileri döndür
        return [t for t in self.takipciler if t.state == TakipDurumu.CONFIRMED and t.time_since_update == 0]

    def _esle(
        self,
        tespitler: List[np.ndarray],
        embeddings: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Maliyet Matrisi ve Macar Algoritması Eşleme Mantığı."""
        if len(self.takipciler) == 0:
            return [], [], list(range(len(tespitler)))
        if len(tespitler) == 0:
            return [], list(range(len(self.takipciler))), []

        n_takip = len(self.takipciler)
        n_tespit = len(tespitler)
        maliyet_matrisi = np.zeros((n_takip, n_tespit), dtype=np.float32)

        for i, t in enumerate(self.takipciler):
            # A. Re-ID Galeri Minimum Kosinüs Mesafesi
            t_features = np.array(t.features)
            d_cos = ReIDEmbeddingCikarici.kosinus_mesafesi(t_features, embeddings)
            min_cos_mesafesi = np.min(d_cos, axis=0)

            # B. Mahalanobis Kapılama (Gating)
            maha_kare = self.kalman.mahalanobis_mesafesi(t.mean, t.covariance, np.array(tespitler))

            for j in range(n_tespit):
                # Eğer Mahalanobis eşiği aşılırsa veya Re-ID mesafesi yüksekse eşlemeyi yasakla (Inf)
                if maha_kare[j] > KalmanKutuFiltresi.MAHALANOBIS_ESIK_095 or min_cos_mesafesi[j] > self.max_cosine_distance:
                    maliyet_matrisi[i, j] = 1e5
                else:
                    maliyet_matrisi[i, j] = min_cos_mesafesi[j]

        # Macar (Hungarian) Algoritması
        satir_ind, sutun_ind = linear_sum_assignment(maliyet_matrisi)

        eslesmeler = []
        eslesmeyen_takipciler = list(range(n_takip))
        eslesmeyen_tespitler = list(range(n_tespit))

        for r, c in zip(satir_ind, sutun_ind):
            if maliyet_matrisi[r, c] < 1e4:
                eslesmeler.append((r, c))
                if r in eslesmeyen_takipciler:
                    eslesmeyen_takipciler.remove(r)
                if c in eslesmeyen_tespitler:
                    eslesmeyen_tespitler.remove(c)

        return eslesmeler, eslesmeyen_takipciler, eslesmeyen_tespitler
