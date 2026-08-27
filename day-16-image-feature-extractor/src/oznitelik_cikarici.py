"""Geleneksel Görsel Öznitelik Çıkarım Modülü (SIFT, ORB, HOG, LBP).

Bu modül; klasik bilgisayarlı görünün en temel 4 öznitelik çıkarıcısını sunar:
1. SIFT: Ölçek ve döndürme bağımsız 128 boyutlu float anahtar nokta betimleyicisi.
2. ORB: Gerçek zamanlı ikili (binary) 256-bit Hamming mesafeli betimleyici.
3. HOG: Nesne sınır ve kenar yönelim dağılımını çıkaran yoğun gradyan histogramı.
4. LBP: Yüz ve malzeme dokularını yerel ikili komşuluklarla kodlayan doku parmak izi.
"""

from dataclasses import dataclass
from typing import Tuple, List
import time
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


@dataclass
class OznitelikOzeti:
    """Bir öznitelik çıkarıcının performans ve boyut metrikleri."""

    algoritma: str
    anahtar_nokta_sayisi: int
    tanimlayici_boyutu: Tuple[int, ...]
    veri_tipi: str
    bellek_bayt: int
    calisma_suresi_ms: float
    aciklama: str


class GorselOznitelikCikarici:
    """SIFT, ORB, HOG ve LBP özniteliklerini hesaplayan birleşik araç seti."""

    @staticmethod
    def sift_cikar(
        gorsel_gri: np.ndarray,
        maks_nokta: int = 500
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray, float]:
        """SIFT (Scale-Invariant Feature Transform) ile 128B float anahtar noktalar çıkarır."""
        if gorsel_gri.ndim != 2:
            raise ValueError("Öznitelik çıkarımı için tek kanallı gri görüntü gereklidir.")

        sift = cv2.SIFT_create(nfeatures=int(maks_nokta))
        t0 = time.perf_counter()
        anahtar_noktalar, tanimlayicilar = sift.detectAndCompute(gorsel_gri, None)
        sure_ms = (time.perf_counter() - t0) * 1000.0

        kp_listesi = list(anahtar_noktalar) if anahtar_noktalar is not None else []
        if tanimlayicilar is None or len(kp_listesi) == 0:
            tanimlayicilar = np.empty((0, 128), dtype=np.float32)

        return kp_listesi, tanimlayicilar, round(sure_ms, 2)

    @staticmethod
    def orb_cikar(
        gorsel_gri: np.ndarray,
        maks_nokta: int = 500
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray, float]:
        """ORB (Oriented FAST and Rotated BRIEF) ile 256-bit (32 byte) ikili tanımlayıcılar çıkarır."""
        if gorsel_gri.ndim != 2:
            raise ValueError("Öznitelik çıkarımı için tek kanallı gri görüntü gereklidir.")

        orb = cv2.ORB_create(nfeatures=int(maks_nokta))
        t0 = time.perf_counter()
        anahtar_noktalar, tanimlayicilar = orb.detectAndCompute(gorsel_gri, None)
        sure_ms = (time.perf_counter() - t0) * 1000.0

        kp_listesi = list(anahtar_noktalar) if anahtar_noktalar is not None else []
        if tanimlayicilar is None or len(kp_listesi) == 0:
            tanimlayicilar = np.empty((0, 32), dtype=np.uint8)

        return kp_listesi, tanimlayicilar, round(sure_ms, 2)

    @staticmethod
    def hog_cikar(
        gorsel_gri: np.ndarray,
        hucre_boyutu: Tuple[int, int] = (8, 8),
        blok_boyutu: Tuple[int, int] = (2, 2),
        yon_kutusu_sayisi: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """HOG (Histogram of Oriented Gradients) vektörü ve görselleştirme haritasını üretir."""
        if gorsel_gri.ndim != 2:
            raise ValueError("HOG için tek kanallı gri görüntü gereklidir.")

        t0 = time.perf_counter()
        vektor, gorsel_harita = hog(
            gorsel_gri,
            orientations=yon_kutusu_sayisi,
            pixels_per_cell=hucre_boyutu,
            cells_per_block=blok_boyutu,
            visualize=True,
            feature_vector=True
        )
        sure_ms = (time.perf_counter() - t0) * 1000.0

        return vektor.astype(np.float32), gorsel_harita, round(sure_ms, 2)

    @staticmethod
    def lbp_cikar(
        gorsel_gri: np.ndarray,
        yari_cap: int = 1,
        nokta_sayisi: int = 8
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """LBP (Local Binary Patterns) doku haritası ve normalize edilmiş histogramını çıkarır."""
        if gorsel_gri.ndim != 2:
            raise ValueError("LBP için tek kanallı gri görüntü gereklidir.")

        t0 = time.perf_counter()
        lbp_harita = local_binary_pattern(
            gorsel_gri,
            P=nokta_sayisi,
            R=yari_cap,
            method="uniform"
        )
        # Uniform LBP için bin sayısı: P + 2
        kutu_sayisi = nokta_sayisi + 2
        histogram, _ = np.histogram(
            lbp_harita.ravel(),
            bins=kutu_sayisi,
            range=(0, kutu_sayisi),
            density=True
        )
        sure_ms = (time.perf_counter() - t0) * 1000.0

        return lbp_harita.astype(np.uint8), histogram.astype(np.float32), round(sure_ms, 2)
