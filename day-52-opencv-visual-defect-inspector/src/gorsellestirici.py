"""
6-Panelli Görsel Kusur ve Bulanıklık Teşhis Panosu (Visual Defect & Blur Inspector Dashboard).
"""

from typing import Dict, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class KusurTeftisGorsellestirici:
    """Bulanıklık, FFT spektrumu ve morfolojik kusur çıktılarını 6 panelli panoda görselleştirir."""

    @classmethod
    def panel_ciz(
        cls,
        bulaniklik_sonuc: Dict[str, Any],
        kusur_sonuc: Dict[str, Any],
        hedef_path: str = "ciktilar/kusur_teftis_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(20, 13), dpi=300)
        fig.suptitle(
            "Day 52: OpenCV ile Kural Tabanlı Görsel Kusur & Bulanıklık Tespiti (AOI Defect Inspector)",
            fontsize=15, fontweight="bold", y=0.98
        )

        lap_var = bulaniklik_sonuc["laplacian_varyansi"]
        fft_hfr = bulaniklik_sonuc["fft_yuksek_frekans_orani"]
        kusur_sayisi = kusur_sonuc["kusur_sayisi"]
        kalite = kusur_sonuc["kalite_puani"]
        netlik_karar = bulaniklik_sonuc["karar"]

        genel_durum = "ONAYLANDI (PASS)" if (netlik_karar == "NET" and kusur_sayisi == 0) else "KUSURLU / AYIKLANMALI (REJECT)"
        kart_renk = "#2ecc71" if genel_durum.startswith("ONAYLANDI") else "#e74c3c"

        # -------------------------------------------------------------
        # Panel 1: Yönetici Karar Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")

        kart_metni = (
            f"AOI TEFTİŞ YÖNETİCİ KARTI\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Laplacian Varyansı: {lap_var:.2f} (Durum: {netlik_karar})\n"
            f"• FFT Yüksek Frekans: %{fft_hfr:.2f}\n"
            f"• Tenengrad Skoru   : {bulaniklik_sonuc['tenengrad_skoru']:.2f}\n"
            f"• Tespit Edilen Kusur: {kusur_sayisi} Adet\n"
            f"• Kusur Alan Oranı  : %{kusur_sonuc['kusur_orani_yuzde']:.3f}\n"
            f"• Kalite Puanı      : {kalite:.1f} / 100\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Genel Karar       : {genel_durum}"
        )

        ax1.text(
            0.5, 0.5, kart_metni, transform=ax1.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.9", facecolor=kart_renk, alpha=0.22, edgecolor=kart_renk, linewidth=2),
            fontsize=9.2, fontweight="bold", family="monospace"
        )
        ax1.set_title("1. AOI Teftiş ve Kalite Özeti", fontweight="bold", color="#2c3e50")

        # -------------------------------------------------------------
        # Panel 2: Orijinal Görüntü & Tespit Edilen Kusur Kutuları
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.imshow(kusur_sonuc["anotasyonlu_gorsel"])
        ax2.axis("off")
        ax2.set_title(f"2. Kusur Tespiti & Konumlandırma ({kusur_sayisi} Kusur)", fontweight="bold", color="#c0392b")

        # -------------------------------------------------------------
        # Panel 3: Laplacian İkinci Dereceden Türev Haritası
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.imshow(bulaniklik_sonuc["laplacian_haritasi"], cmap="inferno")
        ax3.axis("off")
        ax3.set_title(f"3. Laplacian Kenar Haritası (Var={lap_var:.1f})", fontweight="bold", color="#8e44ad")

        # -------------------------------------------------------------
        # Panel 4: 2D FFT Logaritmik Büyüklük Spektrumu
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.imshow(bulaniklik_sonuc["fft_spektrum"], cmap="magma")
        ax4.axis("off")
        ax4.set_title(f"4. 2D FFT Frekans Spektrumu (HFR=%{fft_hfr:.2f})", fontweight="bold", color="#d35400")

        # -------------------------------------------------------------
        # Panel 5: Morfolojik Kusur İkili Maskesi (Binary Defect Mask)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.imshow(kusur_sonuc["binary_mask"], cmap="gray")
        ax5.axis("off")
        ax5.set_title("5. Morfolojik Kusur Maskesi (Top-Hat + Black-Hat)", fontweight="bold", color="#27ae60")

        # -------------------------------------------------------------
        # Panel 6: Kusur Tipleri ve Kalite Dağılımı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        tip_sayilari = {}
        for k in kusur_sonuc["kusurlar"]:
            tip = k["tip"]
            tip_sayilari[tip] = tip_sayilari.get(tip, 0) + 1

        if not tip_sayilari:
            tip_sayilari = {"KUSURSUZ": 1}

        labels = list(tip_sayilari.keys())
        values = list(tip_sayilari.values())
        colors = ["#2ecc71" if l == "KUSURSUZ" else "#e74c3c" for l in labels]

        bars = ax6.bar(labels, values, color=colors, width=0.5, edgecolor="black", linewidth=1)
        ax6.set_ylabel("Tespit Adedi")
        ax6.set_title("6. Kusur Tipi Ayrımı ve Sınıflandırma", fontweight="bold", color="#2c3e50")
        ax6.tick_params(axis="x", rotation=15)

        for bar in bars:
            h = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2., h + 0.05, f"{int(h)}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.32, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
