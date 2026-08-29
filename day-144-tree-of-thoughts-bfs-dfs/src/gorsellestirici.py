"""
Tree of Thoughts (ToT) Teşhis Panosu Görselleştirici Modülü (Day 144 - Faz 8).
6 panelli Game of 24 Yöntem Kıyası, Arama Ağacı ve Budama, BFS vs DFS, Düğüm Dağılımı ve Şema.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class TreeOfThoughtsGorsellestirici:
    """Tree of Thoughts arama ağacı teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        bfs_sonucu: Dict[str, Any],
        dfs_sonucu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/tree_of_thoughts_bfs_dfs_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle(
            "GÜN 144: Tree of Thoughts (ToT): BFS ve DFS Arama ile Düşünce Ağacı Gezintisi & Geri İzleme (Backtracking)",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Game of 24 Yöntemler Kıyası
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        yontemler = ["Input-Output\n(Doğrudan)", "Chain-of-Thought\n(Standart CoT)", "CoT-SC\n(Self-Consistency)", "Tree of Thoughts\n(ToT - BFS/DFS)"]
        basarilar = [7.3, 7.3, 9.0, 78.0]
        renkler1 = ["#e74a3b", "#f6c23e", "#4e73df", "#1cc88a"]

        barlar1 = ax1.bar(yontemler, basarilar, color=renkler1, edgecolor="black", width=0.5)
        for bar in barlar1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

        ax1.set_title("1. Game of 24 Problemi Başarım Kıyaslaması", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Başarı Oranı (%)")
        ax1.set_ylim(0, 95)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: BFS vs DFS Keşfedilen & Budanan Düğümler
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        metrikler = ["Toplam Keşfedilen", "Budanan (Pruned)"]
        bfs_vals = [bfs_sonucu["toplam_kesfedilen_dugum"], bfs_sonucu["toplam_budanan_dugum"]]
        dfs_vals = [dfs_sonucu["toplam_kesfedilen_dugum"], dfs_sonucu["toplam_budanan_dugum"]]

        x = np.arange(len(metrikler))
        w = 0.35

        ax2.bar(x - w / 2, bfs_vals, width=w, label="ToT (BFS)", color="#36b9cc", edgecolor="black")
        ax2.bar(x + w / 2, dfs_vals, width=w, label="ToT (DFS + Backtrack)", color="#f6c23e", edgecolor="black")

        for i in range(len(metrikler)):
            ax2.text(x[i] - w / 2, bfs_vals[i] + 1.0, f"{bfs_vals[i]}", ha="center", fontsize=10, fontweight="bold")
            ax2.text(x[i] + w / 2, dfs_vals[i] + 1.0, f"{dfs_vals[i]}", ha="center", fontsize=10, fontweight="bold")

        ax2.set_title("2. Düğüm Arama ve Budama Sayıları", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Düğüm Sayısı")
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrikler, fontsize=10.5)
        ax2.set_ylim(0, max(bfs_vals + dfs_vals) * 1.25)
        ax2.legend(loc="upper left")
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Budama Verimliliği Oranı
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        budama_orani_bfs = (bfs_sonucu["toplam_budanan_dugum"] / max(1, bfs_sonucu["toplam_kesfedilen_dugum"])) * 100.0
        gecerli_orani_bfs = 100.0 - budama_orani_bfs

        pasta_verileri = [gecerli_orani_bfs, budama_orani_bfs]
        etiketler3 = ["Genişletilen\nYollar", "Budanan Çıkmazlar\n(Pruned)"]
        renkler3 = ["#1cc88a", "#e74a3b"]

        ax3.pie(pasta_verileri, labels=etiketler3, autopct="%1.1f%%", colors=renkler3, startangle=140, explode=(0.05, 0.05))
        ax3.set_title("3. Arama Uzayı Budama (Pruning) Verimliliği", fontsize=12, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 4: Bulunan Çözüm Yolu Adımları
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. Game of 24 [4, 9, 10, 13] Çözüm İzi", fontsize=12, fontweight="bold", pad=10)

        adimlar_metni = "====================================================\n"
        adimlar_metni += "       ToT TARAFINDAN BULUNAN KANITLANMIŞ YOL       \n"
        adimlar_metni += "====================================================\n"
        for i, adim in enumerate(bfs_sonucu["adim_gecmisi"], start=1):
            adimlar_metni += f"  Adım {i}: {adim}\n"
        adimlar_metni += "----------------------------------------------------\n"
        adimlar_metni += f"  Nihai Hedef: {bfs_sonucu['nihai_sayi']} (TAM ÇÖZÜM!)\n"
        adimlar_metni += "===================================================="

        ax4.text(
            0.02, 0.5, adimlar_metni,
            fontsize=9.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: Tree of Thoughts Mimari Akış Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Tree of Thoughts Arama ve Backtrack Şeması", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "        TREE OF THOUGHTS (ToT) FRAMEWORK            \n"
            "====================================================\n"
            "               [Kök Durum: 4 9 10 13]\n"
            "                 ┌───────┴───────┐\n"
            "                 ▼               ▼\n"
            "          [13-9=4, 4, 10]   [10*4=40, 9, 13]\n"
            "          Puan: 0.95        Puan: 0.10 (BUDANDI!)\n"
            "                 │\n"
            "                 ▼\n"
            "          [10-4=6, 4] ──► Puan: 0.95\n"
            "                 │\n"
            "                 ▼\n"
            "          [6 * 4 = 24] ──► HEDEF BULUNDU! (1.00)\n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, sema_metni,
            fontsize=8.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 144 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 144 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "         DAY 144 SUMMARY: TREE OF THOUGHTS (ToT)    \n"
            "====================================================\n"
            "• Game of 24 Başarısı  : %78.0 (CoT: %7.3 vs ToT: %78.0)\n"
            "• Arama Algoritmaları  : BFS (Beam) & DFS (Backtracking)\n"
            "• Durum Değerlendirici : Değer Fonksiyonu V(s) & Budama\n"
            "• Çözüm Adımları       : 13-9=4 -> 10-4=6 -> 6*4=24\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. Doğrusal CoT sınırlarını aşıp ağaç araması yapma\n"
            "  2. Çıkmaz sokakları erkenden tespit edip budama\n"
            "  3. Geri izleme (Backtracking) ile alternatif dallara dönme\n"
            "  4. Planlama ve matematik bulmacalarında devasa başarı\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 145 (Process Reward Models - PRM)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, ozet_metin,
            fontsize=8.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Tree of Thoughts Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
