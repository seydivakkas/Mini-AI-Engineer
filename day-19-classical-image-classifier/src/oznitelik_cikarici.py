"""Geleneksel Görsel Öznitelik Çıkarıcı Modülü.

Görsellerden HOG (Histogram of Oriented Gradients), LBP (Local Binary Patterns)
ve Renk İstatistiklerini çıkararak hibrit öznitelik vektörü üretir.
"""

from typing import Tuple
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


class KlasikOznitelikCikarici:
    """Görsellerden HOG, LBP ve Renk özniteliklerini çıkaran sınıf."""

    def __init__(
        self,
        hedef_boyut: Tuple[int, int] = (64, 64),
        hog_yonelim_sayisi: int = 8,
        hog_hucre_boyutu: Tuple[int, int] = (16, 16),
        hog_blok_boyutu: Tuple[int, int] = (2, 2),
        lbp_nokta_sayisi: int = 8,
        lbp_yaricap: int = 1,
    ) -> None:
        """Öznitelik çıkarıcı parametrelerini yapılandırır.

        Args:
            hedef_boyut: Görsellerin yeniden boyutlandırılacağı (genişlik, yükseklik).
            hog_yonelim_sayisi: HOG açı yönelim bölme sayısı (bin).
            hog_hucre_boyutu: Her HOG hücresinin piksel boyutu (H, W).
            hog_blok_boyutu: Her bloğun kaç hücre içereceği.
            lbp_nokta_sayisi: LBP için komşu örnek nokta sayısı.
            lbp_yaricap: LBP için dairesel komşuluk yarıçapı.
        """
        self.hedef_boyut = hedef_boyut
        self.hog_yonelim_sayisi = hog_yonelim_sayisi
        self.hog_hucre_boyutu = hog_hucre_boyutu
        self.hog_blok_boyutu = hog_blok_boyutu
        self.lbp_nokta_sayisi = lbp_nokta_sayisi
        self.lbp_yaricap = lbp_yaricap

    def _hog_cikar(self, gri: np.ndarray) -> np.ndarray:
        """Görselden HOG (Histogram of Oriented Gradients) özniteliklerini çıkarır."""
        vektor = hog(
            gri,
            orientations=self.hog_yonelim_sayisi,
            pixels_per_cell=self.hog_hucre_boyutu,
            cells_per_block=self.hog_blok_boyutu,
            block_norm="L2-Hys",
            transform_sqrt=True,
            feature_vector=True,
        )
        return vektor.astype(np.float32)

    def _lbp_cikar(self, gri: np.ndarray) -> np.ndarray:
        """Görselden Uniform LBP (Local Binary Pattern) histogramını çıkarır."""
        lbp = local_binary_pattern(
            gri, self.lbp_nokta_sayisi, self.lbp_yaricap, method="uniform"
        )
        n_bins = self.lbp_nokta_sayisi + 2
        hist, _ = np.histogram(
            lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
        )
        return hist.astype(np.float32)

    @staticmethod
    def _renk_momentleri_cikar(img_bgr: np.ndarray) -> np.ndarray:
        """BGR ve HSV uzaylarında ortalama ve standart sapma momentlerini çıkarır (12 boyut)."""
        bgr_mean = np.mean(img_bgr, axis=(0, 1))
        bgr_std = np.std(img_bgr, axis=(0, 1))

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hsv_mean = np.mean(hsv, axis=(0, 1))
        hsv_std = np.std(hsv, axis=(0, 1))

        # 0-1 aralığına normalize et
        momentler = np.concatenate([
            bgr_mean / 255.0,
            bgr_std / 128.0,
            hsv_mean / np.array([180.0, 255.0, 255.0]),
            hsv_std / np.array([90.0, 128.0, 128.0]),
        ])
        return momentler.astype(np.float32)

    def cikar(self, img_bgr: np.ndarray) -> np.ndarray:
        """Girdi görselinden HOG + LBP + Renk Momentleri içeren birleşik vektör üretir.

        Args:
            img_bgr: 3 kanallı (H, W, 3) BGR görseli.

        Returns:
            np.ndarray: Birleştirilmiş öznitelik vektörü.

        Raises:
            ValueError: Görsel boşsa veya 3 kanallı değilse.
        """
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Girdi görseli boş veya geçersiz!")
        if len(img_bgr.shape) != 3 or img_bgr.shape[2] != 3:
            raise ValueError(f"Görsel 3 kanallı BGR olmalıdır. Mevcut: {img_bgr.shape}")

        yeniden_boyutlu = cv2.resize(
            img_bgr, self.hedef_boyut, interpolation=cv2.INTER_AREA
        )
        gri = cv2.cvtColor(yeniden_boyutlu, cv2.COLOR_BGR2GRAY)

        hog_ozn = self._hog_cikar(gri)
        lbp_ozn = self._lbp_cikar(gri)
        renk_ozn = self._renk_momentleri_cikar(yeniden_boyutlu)

        return np.concatenate([hog_ozn, lbp_ozn, renk_ozn])
