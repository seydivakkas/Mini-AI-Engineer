"""Görsel Öznitelik Analiz ve Karşılaştırma Görselleştiricisi (Headless Matplotlib).

Bu modül; SIFT, ORB, HOG ve LBP algoritmalarının anahtar nokta, gradyan
ve doku haritalarını 4 panelli (2x2) karşılaştırmalı bir çizelge halinde kaydeder.
"""

from pathlib import Path
from typing import List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2


class OznitelikGorsellestirici:
    """SIFT, ORB, HOG ve LBP çıktılarını görselleştiren araç."""

    @classmethod
    def analiz_paneli_ciz(
        cls,
        gorsel_gri: np.ndarray,
        sift_kp: List[cv2.KeyPoint],
        orb_kp: List[cv2.KeyPoint],
        hog_harita: np.ndarray,
        lbp_harita: np.ndarray,
        lbp_hist: np.ndarray,
        dosya_yolu: Path
    ) -> Path:
        """4 panelli öznitelik karşılaştırma çizelgesini oluşturup kaydeder."""
        fig, eksenler = plt.subplots(2, 2, figsize=(14, 12), dpi=150)

        # 1. Panel: SIFT Zengin Anahtar Noktaları (Ölçek ve Açı Daireleri)
        sift_cizim = cv2.drawKeypoints(
            gorsel_gri, sift_kp, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            color=(0, 255, 255)
        )
        eksenler[0, 0].imshow(sift_cizim)
        eksenler[0, 0].set_title(
            f"1. SIFT: Ölçek ve Açı Bağımsız Noktalar ({len(sift_kp)} Adet)\n(128-B Boyutlu Float Tanımlayıcı)",
            fontsize=11, fontweight="bold"
        )
        eksenler[0, 0].axis("off")

        # 2. Panel: ORB Anahtar Noktaları (FAST Köşeleri & Oryantasyon)
        orb_cizim = cv2.drawKeypoints(
            gorsel_gri, orb_kp, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            color=(0, 255, 0)
        )
        eksenler[0, 1].imshow(orb_cizim)
        eksenler[0, 1].set_title(
            f"2. ORB: Hızlı İkili (Binary) Noktalar ({len(orb_kp)} Adet)\n(256-Bit / 32 Byte Hamming Tanımlayıcı)",
            fontsize=11, fontweight="bold"
        )
        eksenler[0, 1].axis("off")

        # 3. Panel: HOG Gradyan Yönelimleri Haritası
        eksenler[1, 0].imshow(hog_harita, cmap="inferno")
        eksenler[1, 0].set_title(
            "3. HOG: Gradyan Yönelim Histogramı Haritası\n(Kenar ve Şekil Dağılımı)",
            fontsize=11, fontweight="bold"
        )
        eksenler[1, 0].axis("off")

        # 4. Panel: LBP Doku Haritası
        im_lbp = eksenler[1, 1].imshow(lbp_harita, cmap="magma")
        eksenler[1, 1].set_title(
            f"4. LBP: Yerel İkili Doku Haritası (P=8, R=1)\n(Histogram Kutu Sayısı: {len(lbp_hist)})",
            fontsize=11, fontweight="bold"
        )
        eksenler[1, 1].axis("off")
        fig.colorbar(im_lbp, ax=eksenler[1, 1], fraction=0.046, pad=0.04)

        fig.tight_layout()
        dosya_yolu.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dosya_yolu)
        plt.close(fig)
        return dosya_yolu
