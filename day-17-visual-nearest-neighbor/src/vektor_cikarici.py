"""Görsel Vektör Gömme (Image Embedding) Çıkarıcı Modülü.

Bu modül; herhangi bir görselden renk (HSV 2D histogram), mikro doku (Uniform LBP)
ve geometrik şekil (HOG) özniteliklerini çıkarıp birleştiren ve L2 normalizasyonu
uygulayarak birim küre üzerinde yüksek başarımlı hibrit öznitelik vektörü üreten araçtır.
"""

from typing import Tuple
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


class GorselVektorCikarici:
    """Görüntülerden çok modaliteli (renk + doku + şekil) hibrit öznitelik vektörü çıkarır."""

    def __init__(
        self,
        standart_boyut: Tuple[int, int] = (128, 128),
        agirlik_renk: float = 1.0,
        agirlik_doku: float = 1.0,
        agirlik_sekil: float = 1.0
    ) -> None:
        self.standart_boyut = standart_boyut
        self.agirlik_renk = agirlik_renk
        self.agirlik_doku = agirlik_doku
        self.agirlik_sekil = agirlik_sekil

    def vektor_cikar(self, gorsel_bgr: np.ndarray) -> np.ndarray:
        """Görselden L2 normalize edilmiş 1D hibrit öznitelik vektörü üretir.

        Parametreler:
            gorsel_bgr (np.ndarray): H x W x 3 BGR veya H x W gri görüntü.

        Döndürür:
            np.ndarray: float32 tipinde, L2 normu 1.0 olan 1D vektör.
        """
        if gorsel_bgr is None or gorsel_bgr.size == 0:
            raise ValueError("Girdi görüntüsü boş olamaz.")

        # 3 Kanallı BGR formata standartlaştır
        if gorsel_bgr.ndim == 2:
            gorsel_bgr = cv2.cvtColor(gorsel_bgr, cv2.COLOR_GRAY2BGR)

        # Sabit boyuta ölçekle
        gorsel_islenen = cv2.resize(gorsel_bgr, self.standart_boyut, interpolation=cv2.INTER_AREA)
        gorsel_gri = cv2.cvtColor(gorsel_islenen, cv2.COLOR_BGR2GRAY)

        # 1. Renk Dağılımı: HSV 2D Histogram (Hue x Saturation -> 8 x 8 = 64)
        hsv = cv2.cvtColor(gorsel_islenen, cv2.COLOR_BGR2HSV)
        hist_renk = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).flatten()
        norm_renk = np.linalg.norm(hist_renk)
        if norm_renk > 1e-7:
            hist_renk = hist_renk / norm_renk

        # 2. Mikro Doku: Uniform LBP (P=8, R=1 -> 10 Kutu)
        lbp = local_binary_pattern(gorsel_gri, P=8, R=1, method="uniform")
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
        norm_lbp = np.linalg.norm(hist_lbp)
        if norm_lbp > 1e-7:
            hist_lbp = hist_lbp / norm_lbp

        # 3. Geometrik Şekil ve Kenar: HOG Gradyan Dağılımı
        hog_vektor = hog(
            gorsel_gri,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            feature_vector=True
        )
        norm_hog = np.linalg.norm(hog_vektor)
        if norm_hog > 1e-7:
            hog_vektor = hog_vektor / norm_hog

        # Ağırlıklandır ve Uç Uca Birleştir (Concatenate)
        hibrit_vektor = np.concatenate([
            hist_renk * self.agirlik_renk,
            hist_lbp * self.agirlik_doku,
            hog_vektor * self.agirlik_sekil
        ]).astype(np.float32)

        # L2 Normalizasyonu (Birim Küre İzdüşümü)
        norm = np.linalg.norm(hibrit_vektor)
        if norm > 1e-7:
            hibrit_vektor = hibrit_vektor / norm

        return hibrit_vektor
