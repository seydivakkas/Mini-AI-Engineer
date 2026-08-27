"""Görsel Arama Sonuçları Görselleştiricisi (Headless Matplotlib).

Bu modül; sorgu görselini ve k-NN algoritması tarafından bulunan en yakın Top-K
eşleşmesini etiketleri, mesafeleri ve benzerlik oranlarıyla birlikte karşılaştırmalı
bir galeri çizelgesi olarak diske kaydeder.
"""

from pathlib import Path
from typing import List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.knn_arama_motoru import AramaSonucu


class AramaGorsellestirici:
    """Görsel arama sorgu ve sonuçlarını görselleştiren sunum aracı."""

    @classmethod
    def arama_raporu_ciz(
        cls,
        sorgu_gorseli_bgr: np.ndarray,
        sonuclar: List[AramaSonucu],
        kullanilan_metrik: str,
        dosya_yolu: Path
    ) -> Path:
        """Sorgu görseli ve bulunan Top-K eşleşmeleri tek bir panelde birleştirir."""
        k = len(sonuclar)
        toplam_sutun = k + 1  # 1 Sorgu + K Sonuç

        fig, eksenler = plt.subplots(1, toplam_sutun, figsize=(3.2 * toplam_sutun, 4.2), dpi=140)

        # 1. Sütun: Sorgu Görseli
        eksenler[0].imshow(sorgu_gorseli_bgr[:, :, ::-1])
        eksenler[0].set_title("[SORGU GÖRSELİ]\n(Hedef Arama)", fontsize=11, fontweight="bold", color="blue")
        eksenler[0].axis("off")
        # Mavi çerçeve
        for spine in eksenler[0].spines.values():
            spine.set_edgecolor("blue")
            spine.set_linewidth(3)
            spine.set_visible(True)

        # Sonraki Sütunlar: Top-K Eşleşmeler
        for i, res in enumerate(sonuclar, start=1):
            eksen = eksenler[i]
            eksen.imshow(res.gorsel_bgr[:, :, ::-1])

            renk = "green" if res.benzerlik_yuzdesi >= 75 else ("darkorange" if res.benzerlik_yuzdesi >= 50 else "red")
            baslik_metni = (
                f"#{res.sira}: {res.etiket}\n"
                f"Mesafe: {res.mesafe:.3f}\n"
                f"Uyum: %{res.benzerlik_yuzdesi:.1f}"
            )
            eksen.set_title(baslik_metni, fontsize=10, fontweight="bold", color=renk)
            eksen.axis("off")

            # Dereceye göre renkli çerçeve
            for spine in eksen.spines.values():
                spine.set_edgecolor(renk)
                spine.set_linewidth(2.5)
                spine.set_visible(True)

        fig.suptitle(
            f"Vektör Benzerliği Tabanlı Görsel Arama Raporu (Metrik: {kullanilan_metrik.upper()})",
            fontsize=13, fontweight="bold", y=1.03
        )
        fig.tight_layout()

        dosya_yolu.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dosya_yolu, bbox_inches="tight")
        plt.close(fig)
        return dosya_yolu
