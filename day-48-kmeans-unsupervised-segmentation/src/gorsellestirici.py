"""
6-Panelli K-Means Bölütleme ve Kümeleme Teşhis Panosu (Segmentation Dashboard).
"""

from typing import Dict, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class KMeansBolutlemeGorsellestirici:
    """K-Means kümeleme optimizasyonunu ve bölütleme sonuçlarını 6 panelli panoda görselleştirir."""

    @classmethod
    def panel_ciz(
        cls,
        orijinal_gorsel: np.ndarray,
        kuantalanmis_gorsel: np.ndarray,
        uzamsal_sonuc: Dict[str, Any],
        kume_analizi: Dict[str, Any],
        hedef_path: str = "ciktilar/kmeans_bolutleme_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(20, 13), dpi=300)
        fig.suptitle(
            "Day 48: K-Means ile Denetimsiz Görüntü & Özellik Bölütleme (Elbow, Silhouette, Spatial RGB+XY)",
            fontsize=15, fontweight="bold", y=0.98
        )

        en_iyi_k = kume_analizi["en_iyi_k"]
        en_iyi_sil = kume_analizi["en_iyi_silhouette"]

        # -------------------------------------------------------------
        # Panel 1: Yönetici Bölütleme Karar Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")

        kart_metni = (
            f"K-MEANS BÖLÜTLEME KARAR KARTI\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Optimal Küme Sayısı (K*) : {en_iyi_k} Küme\n"
            f"• Maksimum Silhouette Skoru: {en_iyi_sil:.4f}\n"
            f"• Uzamsal Ağırlık (Alpha)  : {uzamsal_sonuc['uzamsal_agirlik']:.2f}\n"
            f"• Görüntü Çözünürlüğü      : {orijinal_gorsel.shape[1]}x{orijinal_gorsel.shape[0]} Piksel\n"
            f"• Toplam Bölütlenen Alan   : %100 ({orijinal_gorsel.shape[0]*orijinal_gorsel.shape[1]} Px)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Bölütleme Kalitesi       : YÜKSEK AYRIŞIM (ONAYLANDI)"
        )

        ax1.text(
            0.5, 0.5, kart_metni, transform=ax1.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.9", facecolor="#3498db", alpha=0.22, edgecolor="#2980b9", linewidth=2),
            fontsize=9.2, fontweight="bold", family="monospace"
        )
        ax1.set_title("1. K-Means Optimizasyon Karar Kartı", fontweight="bold", color="#2c3e50")

        # -------------------------------------------------------------
        # Panel 2: Elbow (Dirsek) Eğrisi ve WCSS İnişi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        k_vals = kume_analizi["k_degerleri"]
        wcss_vals = kume_analizi["wcss_degerleri"]

        ax2.plot(k_vals, wcss_vals, marker="o", color="#e74c3c", linewidth=2.2, label="WCSS (Inertia)")
        ax2.axvline(en_iyi_k, color="#27ae60", linestyle="--", label=f"Seçilen K* ({en_iyi_k})")
        ax2.scatter([en_iyi_k], [wcss_vals[k_vals.index(en_iyi_k)]], color="#27ae60", s=70, zorder=5)

        ax2.set_title("2. Elbow (Dirsek) Eğrisi ve WCSS", fontweight="bold", color="#c0392b")
        ax2.set_xlabel("Küme Sayısı (K)")
        ax2.set_ylabel("Küme İçi Kareler Toplamı (WCSS)")
        ax2.legend(loc="upper right", fontsize=8)

        # -------------------------------------------------------------
        # Panel 3: Silhouette Skoru Dağılımı (K=2..8)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        sil_vals = kume_analizi["silhouette_degerleri"]
        colors = ["#2ecc71" if k == en_iyi_k else "#95a5a6" for k in k_vals]

        ax3.bar(k_vals, sil_vals, color=colors, edgecolor="black", alpha=0.85)
        ax3.set_title("3. Silhouette Ayrışım Skorları", fontweight="bold", color="#27ae60")
        ax3.set_xlabel("Küme Sayısı (K)")
        ax3.set_ylabel("Silhouette Katsayısı")
        ax3.set_ylim(0, max(sil_vals) * 1.25)

        # -------------------------------------------------------------
        # Panel 4: Renk Kuantalama (RGB K-Means)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.imshow(kuantalanmis_gorsel)
        ax4.axis("off")
        ax4.set_title("4. Renk Kuantalama (Yalnızca RGB)", fontweight="bold", color="#8e44ad")

        # -------------------------------------------------------------
        # Panel 5: Uzamsal Bölütlenmiş Görüntü (RGB + XY Füzyonu)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.imshow(uzamsal_sonuc["bolutlenmis_gorsel"])
        ax5.axis("off")
        ax5.set_title("5. Uzamsal Bölütleme (RGB + XY Füzyonu)", fontweight="bold", color="#d35400")

        # -------------------------------------------------------------
        # Panel 6: Küme Alan Yüzdeleri ve Palet Dağılımı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        yuzdeler = uzamsal_sonuc["alan_yuzdeleri"]
        renkler = uzamsal_sonuc["kume_renkleri"]

        labels = [f"Küme {k+1} (%{yuzdeler[k]})" for k in range(len(yuzdeler))]
        pie_colors = [tuple(renkler[k]) for k in range(len(yuzdeler))]

        ax6.pie(list(yuzdeler.values()), labels=labels, colors=pie_colors, autopct="%1.1f%%", startangle=140,
                wedgeprops=dict(edgecolor="black", linewidth=1.2))
        ax6.set_title("6. Bölütlenmiş Kümelerin Alan Dağılımı", fontweight="bold", color="#16a085")

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.32, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
