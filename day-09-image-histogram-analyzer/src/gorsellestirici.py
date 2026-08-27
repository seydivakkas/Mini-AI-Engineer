"""Histogram Analizi ve Kontrast Karşılaştırma Grafiği (Headless Matplotlib).

Bu modül; orijinal, global eşitlenmiş ve CLAHE uygulanmış görüntüleri,
bunlara ait piksel histogramlarını ve Kümülatif Dağılım Fonksiyonu (CDF) eğrilerini
aynı panel üzerinde 2x3 bir tabloda disk üzerine PNG olarak kaydeder.
"""

from pathlib import Path
from typing import Dict, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.histogram_motoru import HistogramHesaplayici


class HistogramGorsellestirici:
    """Histogram ve CDF eğrilerini karşılaştırmalı olarak çizen görselleştirici."""

    @staticmethod
    def analiz_raporu_ciz(
        gorseller: Dict[str, np.ndarray],
        dosya_yolu: Path
    ) -> Path:
        """3 görüntüyü ve altlarında histogram + CDF eğrilerini 2x3 ızgarada çizer."""
        fig, eksenler = plt.subplots(2, 3, figsize=(15, 8), dpi=150)

        # Başlıklar ve renkler
        renk_paleti = ["#2b5c8f", "#e67e22", "#27ae60"]

        for idx, (isim, resim) in enumerate(gorseller.items()):
            # 1. Satır: Görüntü Önizlemesi
            ax_resim = eksenler[0, idx]
            if resim.ndim == 2:
                ax_resim.imshow(resim, cmap="gray", vmin=0, vmax=255)
            else:
                rgb = resim[:, :, ::-1] if resim.shape[2] == 3 else resim
                ax_resim.imshow(rgb)

            ax_resim.set_title(isim, fontsize=11, fontweight="bold", pad=8)
            ax_resim.axis("off")

            # 2. Satır: Histogram ve CDF Eğrisi
            ax_hist = eksenler[1, idx]
            gri = resim if resim.ndim == 2 else resim[:, :, 0]
            hist = HistogramHesaplayici.kanal_histogrami(gri)
            cdf = HistogramHesaplayici.kumulatif_dagilim_cdf(hist, normalize_et=True)

            # Histogram Çubukları (Sol Eksen)
            ax_hist.bar(range(256), hist, color=renk_paleti[idx], alpha=0.6, width=1.0, label="Piksel Frekansı")
            ax_hist.set_xlim([0, 255])
            ax_hist.set_xlabel("Piksel Değeri (0 - 255)", fontsize=9)
            ax_hist.set_ylabel("Frekans", fontsize=9, color=renk_paleti[idx])
            ax_hist.grid(True, linestyle="--", alpha=0.3)

            # CDF Eğrisi (İkiz Sağ Eksen)
            ax_cdf = ax_hist.twinx()
            ax_cdf.plot(range(256), cdf, color="#c0392b", linewidth=2, label="CDF Eğrisi")
            ax_cdf.set_ylim([0, 1.05])
            ax_cdf.set_ylabel("Kümülatif Oran (CDF)", fontsize=9, color="#c0392b")

            ax_hist.set_title(f"{isim} - Histogram & CDF", fontsize=10, fontweight="bold")

        fig.suptitle("Görüntü Histogramı ve Kontrast İyileştirme Karşılaştırma Raporu", fontsize=14, fontweight="bold")
        fig.tight_layout()

        dosya_yolu.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dosya_yolu)
        plt.close(fig)
        return dosya_yolu
