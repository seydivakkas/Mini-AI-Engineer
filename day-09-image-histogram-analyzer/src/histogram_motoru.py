"""Görüntü Histogramı ve Kontrast İyileştirme Motoru (CDF, Global Eşitleme, CLAHE).

Bu modül; gri tonlamalı ve renkli görüntülerde piksel yoğunluk histogramlarını,
Kümülatif Dağılım Fonksiyonunu (CDF), renk bozulmasına yol açmayan
Global Histogram Eşitlemeyi ve yerel parazit sınırlandırmalı CLAHE algoritmasını yürütür.
"""

from typing import Dict, Tuple
import cv2
import numpy as np


class HistogramHesaplayici:
    """Piksel frekans dağılımları ve bilgi entropisi analizörü."""

    @staticmethod
    def kanal_histogrami(gorsel: np.ndarray, kanal_indeksi: int = 0) -> np.ndarray:
        """Belirtilen kanal için 256 kutucuklu (bin) 1B histogram hesaplar."""
        if gorsel.ndim == 2:
            resim = [gorsel]
        else:
            resim = [gorsel]

        hist = cv2.calcHist(
            images=resim,
            channels=[kanal_indeksi if gorsel.ndim > 2 else 0],
            mask=None,
            histSize=[256],
            ranges=[0, 256]
        )
        return hist.flatten()

    @classmethod
    def renkli_histogramlar(cls, gorsel_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """Renkli (BGR) görüntünün 3 kanalı için ayrı ayrı histogram çıkarır."""
        if gorsel_bgr.ndim != 3:
            raise ValueError("Renkli histogram için 3 kanallı BGR görüntü gereklidir.")

        return {
            "Mavi": cls.kanal_histogrami(gorsel_bgr, 0),
            "Yeşil": cls.kanal_histogrami(gorsel_bgr, 1),
            "Kırmızı": cls.kanal_histogrami(gorsel_bgr, 2)
        }

    @staticmethod
    def kumulatif_dagilim_cdf(histogram: np.ndarray, normalize_et: bool = True) -> np.ndarray:
        """Histogramın Kümülatif Dağılım Fonksiyonunu (CDF) hesaplar.

        Formül:
            CDF[k] = sum_{j=0}^k H[j]
        """
        cdf = histogram.cumsum()
        if normalize_et:
            maks = cdf[-1]
            if maks > 0:
                cdf = cdf / maks
        return cdf

    @classmethod
    def kontrast_metrikleri(cls, gorsel: np.ndarray) -> Dict[str, float]:
        """Görüntünün kontrastını, dinamik aralığını ve Shannon Entropisini ölçer."""
        gri = gorsel if gorsel.ndim == 2 else cv2.cvtColor(gorsel, cv2.COLOR_BGR2GRAY)

        min_val, max_val = float(np.min(gri)), float(np.max(gri))
        ortalama = float(np.mean(gri))
        std_sapma = float(np.std(gri))  # RMS Kontrast
        dinamik_aralik = max_val - min_val

        # Shannon Entropisi: H = - sum(p * log2(p))
        hist = cls.kanal_histogrami(gri)
        p = hist / (np.sum(hist) + 1e-12)
        p_pozitif = p[p > 0]
        entropi = -float(np.sum(p_pozitif * np.log2(p_pozitif)))

        return {
            "min_piksel": min_val,
            "max_piksel": max_val,
            "ortalama_parlaklik": round(ortalama, 2),
            "rms_kontrast": round(std_sapma, 2),
            "dinamik_aralik": round(dinamik_aralik, 2),
            "shannon_entropisi": round(entropi, 3)
        }


class KontrastIyilestirici:
    """Global Histogram Eşitleme ve CLAHE Dönüştürücü."""

    @staticmethod
    def global_histogram_esitle(gorsel: np.ndarray) -> np.ndarray:
        """Global histogram eşitlemesi uygular.

        Renkli görsellerde renk bozulmasını önlemek için YCrCb uzayında
        yalnızca Aydınlık (Y) kanalı eşitlenir!
        """
        if gorsel.ndim == 2:
            return cv2.equalizeHist(gorsel)

        # Renkli (BGR) görüntü: YCrCb uzayına geçiş
        ycrcb = cv2.cvtColor(gorsel, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def clahe_uygula(
        gorsel: np.ndarray,
        kirpma_limiti: float = 2.0,
        karo_boyutu: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """Kontrast Sınırlı Uyarlanabilir Histogram Eşitleme (CLAHE) uygular.

        Renkli görüntülerde LAB uzayında L (Aydınlık) kanalı işlenir.

        Parametreler:
            kirpma_limiti (float): Gürültü patlamasını engelleyen tepe kırpma sınırı.
            karo_boyutu (Tuple[int, int]): Görüntünün bölüneceği yerel ızgara boyutu.
        """
        clahe = cv2.createCLAHE(clipLimit=kirpma_limiti, tileGridSize=karo_boyutu)

        if gorsel.ndim == 2:
            return clahe.apply(gorsel)

        # Renkli (BGR) görüntü: LAB uzayına geçiş
        lab = cv2.cvtColor(gorsel, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
