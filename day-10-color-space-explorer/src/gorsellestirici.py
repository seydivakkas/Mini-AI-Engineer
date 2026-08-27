"""Renk Uzayları Görselleştirme ve Analiz Paneli (Headless Matplotlib).

Bu modül; RGB, HSV ve LAB renk uzaylarının kanal bazlı ayrışımlarını,
farklı aydınlatma altındaki davranışlarını ve renk segmentasyonu sonucunu
12 panelli zengin bir görsel ızgara olarak disk üzerine kaydeder.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class RenkUzayiGorsellestirici:
    """Renk kanallarını ve segmentasyon sonuçlarını çizen panel motoru."""

    @staticmethod
    def analiz_paneli_ciz(
        orijinal_bgr: np.ndarray,
        rgb_kanallari: Tuple[np.ndarray, np.ndarray, np.ndarray],
        hsv_kanallari: Tuple[np.ndarray, np.ndarray, np.ndarray],
        lab_kanallari: Tuple[np.ndarray, np.ndarray, np.ndarray],
        maske: np.ndarray,
        segmente_bgr: np.ndarray,
        dosya_yolu: Path
    ) -> Path:
        """3x4 = 12 panelli kapsamlı renk uzayı analiz çizelgesini diske kaydeder."""
        fig, eksenler = plt.subplots(3, 4, figsize=(16, 11), dpi=150)

        # 1. Satır: Orijinal ve RGB Kanalları
        # Orijinal BGR -> RGB
        eksenler[0, 0].imshow(orijinal_bgr[:, :, ::-1])
        eksenler[0, 0].set_title("1. Orijinal Görüntü (Gölge Geçişli)", fontsize=9, fontweight="bold")

        eksenler[0, 1].imshow(rgb_kanallari[0], cmap="Reds")
        eksenler[0, 1].set_title("2. Kırmızı Kanalı (R)", fontsize=9, fontweight="bold")

        eksenler[0, 2].imshow(rgb_kanallari[1], cmap="Greens")
        eksenler[0, 2].set_title("3. Yeşil Kanalı (G)", fontsize=9, fontweight="bold")

        eksenler[0, 3].imshow(rgb_kanallari[2], cmap="Blues")
        eksenler[0, 3].set_title("4. Mavi Kanalı (B)", fontsize=9, fontweight="bold")

        # 2. Satır: HSV Kanalları ve LAB L Kanalı
        eksenler[1, 0].imshow(hsv_kanallari[0], cmap="hsv")
        eksenler[1, 0].set_title("5. HSV - Ton (Hue / Saf Renk)", fontsize=9, fontweight="bold")

        eksenler[1, 1].imshow(hsv_kanallari[1], cmap="gray")
        eksenler[1, 1].set_title("6. HSV - Doygunluk (Saturation)", fontsize=9, fontweight="bold")

        eksenler[1, 2].imshow(hsv_kanallari[2], cmap="gray")
        eksenler[1, 2].set_title("7. HSV - Değer (Value / Parlaklık)", fontsize=9, fontweight="bold")

        eksenler[1, 3].imshow(lab_kanallari[0], cmap="gray")
        eksenler[1, 3].set_title("8. LAB - Aydınlık (L* Kanalı)", fontsize=9, fontweight="bold")

        # 3. Satır: LAB Renk Düzlemleri ve Segmentasyon
        eksenler[2, 0].imshow(lab_kanallari[1], cmap="coolwarm")
        eksenler[2, 0].set_title("9. LAB - a* (Yeşil <-> Kırmızı)", fontsize=9, fontweight="bold")

        eksenler[2, 1].imshow(lab_kanallari[2], cmap="coolwarm")
        eksenler[2, 1].set_title("10. LAB - b* (Mavi <-> Sarı)", fontsize=9, fontweight="bold")

        eksenler[2, 2].imshow(maske, cmap="gray")
        eksenler[2, 2].set_title("11. Kırmızı Renk Maskesi", fontsize=9, fontweight="bold")

        eksenler[2, 3].imshow(segmente_bgr[:, :, ::-1])
        eksenler[2, 3].set_title("12. İzole Edilmiş Hedef Nesne", fontsize=9, fontweight="bold")

        for ax in eksenler.ravel():
            ax.axis("off")

        fig.suptitle("Renk Uzayları Analiz ve Gölgeye Dayanıklı Segmentasyon Paneli", fontsize=14, fontweight="bold", y=0.98)
        fig.tight_layout()

        dosya_yolu.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dosya_yolu)
        plt.close(fig)
        return dosya_yolu
