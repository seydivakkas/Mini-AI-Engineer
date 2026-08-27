"""Baskın Renk Paleti ve Kuantizasyon Görselleştirici (Headless Matplotlib).

Bu modül; orijinal görüntüyü, K renge kuantize edilmiş versiyonunu
ve yüzdesel oranlara göre ölçeklenmiş renk paleti şeridini (Color Swatch)
disk üzerine yüksek çözünürlüklü PNG olarak kaydeder.
"""

from pathlib import Path
from typing import List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from src.renk_kumeleyici import RenkBilgisi


class PaletGorsellestirici:
    """Renk paletini ve kuantizasyon sonuçlarını görselleştiren araç."""

    @staticmethod
    def palet_raporu_ciz(
        orijinal_bgr: np.ndarray,
        quantize_bgr: np.ndarray,
        palet: List[RenkBilgisi],
        dosya_yolu: Path
    ) -> Path:
        """Orijinal, kuantize ve renk şeridini içeren görsel rapor oluşturur."""
        fig = plt.figure(figsize=(14, 9), dpi=150)
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.2])

        # 1. Panel: Orijinal Görüntü
        ax_orijinal = fig.add_subplot(gs[0, 0])
        ax_orijinal.imshow(orijinal_bgr[:, :, ::-1])
        ax_orijinal.set_title("Orijinal Görüntü", fontsize=11, fontweight="bold")
        ax_orijinal.axis("off")

        # 2. Panel: Kuantize Edilmiş Görüntü (K Renk)
        ax_quant = fig.add_subplot(gs[0, 1])
        ax_quant.imshow(quantize_bgr[:, :, ::-1])
        ax_quant.set_title(f"Kuantize Edilmiş Görüntü (K = {len(palet)} Renk)", fontsize=11, fontweight="bold")
        ax_quant.axis("off")

        # 3. Panel: Orantısal Renk Paleti Şeridi (Color Swatch Bar)
        ax_palet = fig.add_subplot(gs[1, :])
        ax_palet.set_xlim([0, 100])
        ax_palet.set_ylim([0, 1])
        ax_palet.axis("off")

        mevcut_x = 0.0
        for renk in palet:
            genislik = renk.yuzde
            renk_norm = [c / 255.0 for c in renk.rgb]

            # Renk kutusu
            dikdortgen = patches.Rectangle(
                (mevcut_x, 0.1), genislik, 0.8,
                facecolor=renk_norm, edgecolor="black", linewidth=1.5
            )
            ax_palet.add_patch(dikdortgen)

            # Yazı etiketi (Metin rengi arka plan parlaklığına göre siyah veya beyaz seçilir)
            parlaklik = (renk.rgb[0] * 0.299 + renk.rgb[1] * 0.587 + renk.rgb[2] * 0.114)
            metin_renk = "black" if parlaklik > 130 else "white"

            if genislik >= 6.0:  # Yazı sığacak kadar genişse etiketi yaz
                ax_palet.text(
                    mevcut_x + (genislik / 2.0), 0.5,
                    f"{renk.hex_kodu}\n%{renk.yuzde:.1f}",
                    ha="center", va="center",
                    color=metin_renk, fontsize=9, fontweight="bold"
                )

            mevcut_x += genislik

        ax_palet.set_title("Baskın Renk Paleti ve Ağırlık Dağılımı (%)", fontsize=11, fontweight="bold", pad=8)
        fig.tight_layout()

        dosya_yolu.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dosya_yolu)
        plt.close(fig)
        return dosya_yolu
