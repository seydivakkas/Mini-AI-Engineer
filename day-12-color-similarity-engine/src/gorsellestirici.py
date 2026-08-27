"""Algısal Renk Arama Sonuçları Görselleştirici (Headless Matplotlib).

Bu modül; sorgu görselini ve en benzer Top-K katalog ürününü, her birinin
renk şeritlerini ve benzerlik skoru rozetlerini tek bir karşılaştırma
paneli halinde disk üzerine kaydeder.
"""

from pathlib import Path
from typing import List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from src.palet_eslestirici import PaletRengi
from src.katalog_arama import AramaSonucu


class AramaGorsellestirici:
    """Arama sonuçlarını ve renk paletlerini görselleştiren araç."""

    @staticmethod
    def _palet_seridi_ciz(ax, palet: List[PaletRengi], baslik_metni: str = "") -> None:
        """Belirtilen eksene orantısal renk şeridi çizer."""
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 1])
        ax.axis("off")

        toplam_agirlik = sum(r.agirlik for r in palet)
        mevcut_x = 0.0
        for r in palet:
            genislik = (r.agirlik / toplam_agirlik) * 100.0
            norm_rgb = [c / 255.0 for c in r.rgb]
            dikdortgen = patches.Rectangle(
                (mevcut_x, 0.15), genislik, 0.7,
                facecolor=norm_rgb, edgecolor="black", linewidth=1.0
            )
            ax.add_patch(dikdortgen)

            parlaklik = r.rgb[0] * 0.299 + r.rgb[1] * 0.587 + r.rgb[2] * 0.114
            metin_renk = "black" if parlaklik > 130 else "white"

            if genislik >= 12.0:
                ax.text(
                    mevcut_x + (genislik / 2.0), 0.5,
                    f"{r.hex_kodu}\n%{genislik:.0f}",
                    ha="center", va="center",
                    color=metin_renk, fontsize=7.5, fontweight="bold"
                )
            mevcut_x += genislik

        if baslik_metni:
            ax.set_title(baslik_metni, fontsize=9, fontweight="bold", pad=4)

    @classmethod
    def arama_raporu_ciz(
        cls,
        sorgu_gorsel_bgr: np.ndarray,
        sorgu_paleti: List[PaletRengi],
        sonuclar: List[AramaSonucu],
        dosya_yolu: Path
    ) -> Path:
        """Sorgu ve bulunan en iyi sonuçları karşılaştırmalı görselleştirir."""
        k = len(sonuclar)
        toplam_sutun = 1 + k  # 1 sorgu + K sonuç

        fig = plt.figure(figsize=(4 * toplam_sutun, 7), dpi=150)
        gs = fig.add_gridspec(2, toplam_sutun, height_ratios=[3, 1])

        # 1. Sütun: Sorgu Görseli ve Paleti
        ax_sorgu_img = fig.add_subplot(gs[0, 0])
        ax_sorgu_img.imshow(sorgu_gorsel_bgr[:, :, ::-1])
        ax_sorgu_img.set_title("KULLANICI SORGUSU\n(Referans Görsel)", fontsize=11, fontweight="bold", color="darkblue")
        ax_sorgu_img.axis("off")

        ax_sorgu_palet = fig.add_subplot(gs[1, 0])
        cls._palet_seridi_ciz(ax_sorgu_palet, sorgu_paleti, "Sorgu Paleti")

        # Diğer Sütunlar: Eşleşen Arama Sonuçları
        for idx, sonuc in enumerate(sonuclar, 1):
            ax_img = fig.add_subplot(gs[0, idx])
            ax_img.imshow(sonuc.urun.gorsel_bgr[:, :, ::-1])

            # Benzerlik rozeti rengi
            if sonuc.benzerlik_skoru >= 75.0:
                rozet_renk = "forestgreen"
            elif sonuc.benzerlik_skoru >= 50.0:
                rozet_renk = "darkorange"
            else:
                rozet_renk = "firebrick"

            baslik = f"SIRA #{idx}: {sonuc.urun.ad}\n"
            baslik += f"Benzerlik: %{sonuc.benzerlik_skoru:.1f} (Delta-E: {sonuc.delta_e_mesafesi:.1f})"
            ax_img.set_title(baslik, fontsize=10, fontweight="bold", color=rozet_renk)
            ax_img.axis("off")

            ax_palet = fig.add_subplot(gs[1, idx])
            cls._palet_seridi_ciz(ax_palet, sonuc.urun.palet, f"Ürün Paleti (#{sonuc.urun.urun_id})")

        fig.tight_layout()
        dosya_yolu.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dosya_yolu)
        plt.close(fig)
        return dosya_yolu
