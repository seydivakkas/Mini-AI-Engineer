"""
Halı Görsel Arama ve Çoklu Özellik Füzyonu 6-Panelli Teşhis Panosu.
"""

from typing import Dict, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


class HaliGorselAramaGorsellestirici:
    """Görsel arama sorgusunu, Top-K katalog eşleşmelerini ve öznitelik füzyonunu gösteren pano."""

    @classmethod
    def arama_paneli_ciz(
        cls,
        sorgu_gorseli: Image.Image,
        arama_sonucu: Dict[str, Any],
        hedef_path: str = "ciktilar/hali_gorsel_arama_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(19, 12), dpi=300)
        fig.suptitle(
            "Day 38: Halı Doku ve Desenleri İçin Çoklu Özellikli (Renk + GLCM Doku) Görsel Arama Paneli",
            fontsize=15, fontweight="bold", y=0.98
        )

        top_sonuclar = arama_sonucu.get("sonuclar", [])

        # -------------------------------------------------------------
        # Panel 1: Sorgu Halısı (Query Carpet)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.imshow(sorgu_gorseli)
        h_ist = arama_sonucu["sorgu_ozellikleri"]["haralick"]
        info_text = f"Sorgu Halısı Özellikleri:\n• Kontrast: {h_ist['kontrast']:.2f}\n• Homojenlik: {h_ist['homojenlik']:.2f}\n• Enerji: {h_ist['enerji']:.2f}"
        ax1.set_title("1. Sorgu Halısı (Query Carpet)", fontweight="bold", color="#1f77b4")
        ax1.axis("off")
        ax1.text(
            0.05, 0.05, info_text, transform=ax1.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray"),
            fontsize=8, fontweight="bold"
        )

        # -------------------------------------------------------------
        # Panel 2: En İyi 3 Katalog Eşleşmesi (Top-3 Matches)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        n_top = min(3, len(top_sonuclar))
        if n_top > 0:
            gen = 100
            yuk = 100
            panoramik = np.zeros((yuk, gen * n_top + (n_top - 1) * 10, 3), dtype=np.uint8)

            for i, res in enumerate(top_sonuclar[:n_top]):
                thumb = res["gorsel"].resize((gen, yuk))
                x_off = i * (gen + 10)
                panoramik[:, x_off:x_off + gen] = np.array(thumb)

            ax2.imshow(panoramik)
            ax2.axis("off")

            alt_yazi = " | ".join([f"#{i+1}: %{res['hibrit_skor']:.1f}" for i, res in enumerate(top_sonuclar[:n_top])])
            ax2.set_title(f"2. Top-{n_top} Katalog Eşleşmesi\n({alt_yazi})", fontweight="bold", color="#2ca02c", fontsize=10)
        else:
            ax2.text(0.5, 0.5, "Eşleşme Bulunamadı", ha="center", va="center")
            ax2.axis("off")

        # -------------------------------------------------------------
        # Panel 3: GLCM Haralick Doku Metrikleri Karşılaştırması
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        metrikler = ["kontrast", "homojenlik", "enerji", "korelasyon", "entropi"]
        labels = ["Kontrast", "Homojenlik", "Enerji", "Korelasyon", "Entropi"]
        s_vals = [h_ist[m] for m in metrikler]

        top1 = top_sonuclar[0] if top_sonuclar else None
        t1_vals = [top1["haralick"][m] for m in metrikler] if top1 else s_vals

        x_m = np.arange(len(metrikler))
        w = 0.35
        ax3.bar(x_m - w / 2, s_vals, width=w, label="Sorgu Halısı", color="#3498db", edgecolor="black")
        ax3.bar(x_m + w / 2, t1_vals, width=w, label="Top-1 Eşleşme", color="#e67e22", edgecolor="black")
        ax3.set_xticks(x_m)
        ax3.set_xticklabels(labels, fontsize=8, rotation=15)
        ax3.set_ylabel("Haralick Değeri", fontweight="bold", fontsize=9)
        ax3.set_title("3. GLCM Doku Karakteristiği (Sorgu vs Top-1)", fontweight="bold", color="#d62728")
        ax3.legend(fontsize=8, loc="upper right")

        # -------------------------------------------------------------
        # Panel 4: Renk vs Doku Benzerlik Skor Ayrışımı
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ids = [f"{r['id'][-8:]}\n(%{r['hibrit_skor']:.1f})" for r in top_sonuclar]
        r_skorlar = [r["renk_skor"] for r in top_sonuclar]
        d_skorlar = [r["doku_skor"] for r in top_sonuclar]

        x_ids = np.arange(len(ids))
        w_b = 0.35
        ax4.bar(x_ids - w_b / 2, r_skorlar, width=w_b, label="Renk Skoru", color="#9b59b6", edgecolor="black")
        ax4.bar(x_ids + w_b / 2, d_skorlar, width=w_b, label="Doku Skoru", color="#1abc9c", edgecolor="black")
        ax4.set_xticks(x_ids)
        ax4.set_xticklabels(ids, fontsize=7.5)
        ax4.set_ylabel("Benzerlik Skoru (%)", fontweight="bold", fontsize=9)
        ax4.set_ylim(0, 110)
        ax4.set_title("4. Çoklu Özellik Benzerlik Ayrışımı", fontweight="bold", color="#9467bd")
        ax4.legend(fontsize=8, loc="lower right")

        # -------------------------------------------------------------
        # Panel 5: Arama Ağırlık Füzyon Dağılımı (Donut Chart)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        agirliklar = arama_sonucu["kullanilan_agirliklar"]
        pie_vals = [agirliklar["renk_agirligi"] * 100, agirliklar["doku_agirligi"] * 100]
        pie_labels = [f"Renk Katkısı\n(%{pie_vals[0]:.0f})", f"Doku Katkısı\n(%{pie_vals[1]:.0f})"]
        pie_colors = ["#34495e", "#f39c12"]

        wedges, texts, autotexts = ax5.pie(
            pie_vals, labels=pie_labels, autopct="%1.0f%%",
            startangle=90, colors=pie_colors,
            wedgeprops=dict(width=0.45, edgecolor="black", linewidth=1.2)
        )
        for at in autotexts:
            at.set_fontweight("bold")
            at.set_fontsize(9)
        ax5.set_title("5. Füzyon Ağırlık Dağılımı", fontweight="bold", color="#ff7f0e")

        # -------------------------------------------------------------
        # Panel 6: Katalog Benzerlik Rank & Dağılım Grafiği
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        rank_skorlari = [r["hibrit_skor"] for r in top_sonuclar]
        rank_sirasi = np.arange(1, len(rank_skorlari) + 1)

        ax6.plot(rank_sirasi, rank_skorlari, marker="o", markersize=8, color="#27ae60", linewidth=2.2, label="Hibrit Sıralama")
        ax6.axhline(80.0, color="red", linestyle="--", linewidth=1.2, label="Yüksek Benzerlik Eşiği (%80)")
        ax6.set_xlabel("Rank (Sıralama)", fontweight="bold", fontsize=9)
        ax6.set_ylabel("Hibrit Benzerlik (%)", fontweight="bold", fontsize=9)
        ax6.set_xticks(rank_sirasi)
        ax6.set_ylim(0, 105)
        ax6.set_title("6. Katalog Arama Sıralama Eğrisi", fontweight="bold", color="#333333")
        ax6.legend(fontsize=8, loc="lower left")

        for r, s in zip(rank_sirasi, rank_skorlari):
            ax6.annotate(f"%{s:.1f}", (r, s), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.32, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
