"""Görsel Öznitelik Çıkarıcı Modülü.

Görsellerden renk, doku ve şekil modalitelerini çıkararak dengeli,
L2 normalize edilmiş çok modaliteli embedding vektörleri üretir.
"""

from typing import Tuple
import cv2
import numpy as np
from skimage.feature import local_binary_pattern


class GorselVektorCikarici:
    """Görsellerden hibrit (Renk + Doku + Şekil) öznitelik vektörü çıkaran sınıf."""

    def __init__(
        self,
        hedef_boyut: Tuple[int, int] = (64, 64),
        renk_agirligi: float = 0.40,
        doku_agirligi: float = 0.30,
        sekil_agirligi: float = 0.30,
    ) -> None:
        """Öznitelik çıkarıcı parametrelerini ilklendirir.

        Args:
            hedef_boyut: Görsellerin ölçekleneceği (genişlik, yükseklik).
            renk_agirligi: Hibrit vektördeki renk bileşeni ağırlığı.
            doku_agirligi: Hibrit vektördeki LBP doku bileşeni ağırlığı.
            sekil_agirligi: Hibrit vektördeki şekil/gradyan bileşeni ağırlığı.
        """
        self.hedef_boyut = hedef_boyut
        self.renk_agirligi = renk_agirligi
        self.doku_agirligi = doku_agirligi
        self.sekil_agirligi = sekil_agirligi

        # LBP Parametreleri (Uniform LBP: P=8, R=1 -> 10 boyutlu histogram)
        self.lbp_nokta_sayisi = 8
        self.lbp_yaricap = 1

    @staticmethod
    def _l2_normalize(vektor: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Vektörü L2 normuna göre normalize eder."""
        norm = np.linalg.norm(vektor)
        return vektor / (norm + eps)

    def _renk_ozniteligi_cikar(self, img_bgr: np.ndarray) -> np.ndarray:
        """HSV uzayında 8x8 Hue-Saturation 2D histogramı (64 boyut) çıkarır."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]
        )
        hist = hist.flatten().astype(np.float32)
        return self._l2_normalize(hist)

    def _doku_ozniteligi_cikar(self, gri: np.ndarray) -> np.ndarray:
        """Uniform LBP (Local Binary Pattern) histogramı (10 boyut) çıkarır."""
        lbp = local_binary_pattern(
            gri, self.lbp_nokta_sayisi, self.lbp_yaricap, method="uniform"
        )
        n_bins = self.lbp_nokta_sayisi + 2
        hist, _ = np.histogram(
            lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
        )
        hist = hist.astype(np.float32)
        return self._l2_normalize(hist)

    def _sekil_ozniteligi_cikar(self, gri: np.ndarray) -> np.ndarray:
        """Sobel yatay ve dikey gradyan enerjisi ve kenar yoğunluğu (64 boyut) çıkarır."""
        gx = cv2.Sobel(gri, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gri, cv2.CV_32F, 0, 1, ksize=3)
        mag, _ = cv2.cartToPolar(gx, gy)
        
        # 8x8 uzamsal bloklara bölerek gradyan enerjisi dağılımı (8x8 = 64 boyut)
        h, w = gri.shape
        bh, bw = h // 8, w // 8
        blok_enerjileri = []
        for i in range(8):
            for j in range(8):
                blok = mag[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
                blok_enerjileri.append(np.mean(blok))
                
        sekil_vektor = np.array(blok_enerjileri, dtype=np.float32)
        return self._l2_normalize(sekil_vektor)

    def cikar(self, gorsel_bgr: np.ndarray) -> np.ndarray:
        """Verilen BGR görselden L2 normalize edilmiş hibrit öznitelik vektörü üretir.

        Args:
            gorsel_bgr: Giriş görseli (H, W, 3) uint8 numpy dizisi.

        Returns:
            138 boyutlu (64 Renk + 10 Doku + 64 Şekil) L2 normalize vektör.

        Raises:
            ValueError: Görsel boşsa veya 3 kanallı değilse.
        """
        if gorsel_bgr is None or gorsel_bgr.size == 0:
            raise ValueError("Girdi görseli boş veya geçersiz!")
        if len(gorsel_bgr.shape) != 3 or gorsel_bgr.shape[2] != 3:
            raise ValueError(f"Görsel 3 kanallı BGR olmalıdır. Mevcut şekil: {gorsel_bgr.shape}")

        yeniden_boyutlu = cv2.resize(
            gorsel_bgr, self.hedef_boyut, interpolation=cv2.INTER_AREA
        )
        gri = cv2.cvtColor(yeniden_boyutlu, cv2.COLOR_BGR2GRAY)

        v_renk = self._renk_ozniteligi_cikar(yeniden_boyutlu)
        v_doku = self._doku_ozniteligi_cikar(gri)
        v_sekil = self._sekil_ozniteligi_cikar(gri)

        hibrit = np.concatenate([
            v_renk * self.renk_agirligi,
            v_doku * self.doku_agirligi,
            v_sekil * self.sekil_agirligi,
        ])

        return self._l2_normalize(hibrit)
