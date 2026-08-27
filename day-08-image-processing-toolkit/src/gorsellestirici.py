"""Görüntü İşleme Karşılaştırma ve Panel Üreteci Modülü (Headless Matplotlib).

Bu modül; orijinal görseli, Gauss yumuşatmasını, Sobel kenar gradyanlarını
ve morfolojik dönüşüm adımlarını tek bir yüksek çözünürlüklü karşılaştırma paneli
halinde disk üzerine PNG olarak kaydeder.
"""

from pathlib import Path
from typing import Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class IslemePaneliUreteci:
    """Filtreleme ve morfoloji aşamalarını görsel bir panelde toplayan araç."""

    @staticmethod
    def panel_olustur_ve_kaydet(
        adimlar: Dict[str, np.ndarray],
        dosya_yolu: Path,
        panel_basligi: str = "Temel Görüntü İşleme & Morfoloji Analiz Paneli"
    ) -> Path:
        """Sözlükteki her bir adımı 3x3 bir görsel ızgarada diske kaydeder."""
        toplam_adim = len(adimlar)
        satir = 3
        sutun = 3

        fig, eksenler = plt.subplots(satir, sutun, figsize=(12, 11), dpi=150)
        eksenler_duz = eksenler.ravel()

        for idx, (baslik, resim) in enumerate(adimlar.items()):
            ax = eksenler_duz[idx]
            if resim.ndim == 2:
                ax.imshow(resim, cmap="gray", vmin=0, vmax=255)
            else:
                # BGR -> RGB dönüşümü
                rgb = resim[:, :, ::-1] if resim.shape[2] == 3 else resim
                ax.imshow(rgb)

            ax.set_title(baslik, fontsize=10, fontweight="bold", pad=6)
            ax.axis("off")

        # Artık kalan eksenleri kapat
        for idx in range(toplam_adim, len(eksenler_duz)):
            fig.delaxes(eksenler_duz[idx])

        fig.suptitle(panel_basligi, fontsize=14, fontweight="bold", y=0.98)
        fig.tight_layout()

        dosya_yolu.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dosya_yolu)
        plt.close(fig)
        return dosya_yolu
