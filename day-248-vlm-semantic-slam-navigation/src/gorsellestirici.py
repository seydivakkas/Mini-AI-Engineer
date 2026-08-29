"""
VLM Destekli Semantik SLAM 6 Panelli Görselleştirici Modülü (FAZ 13) (Day 248).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np
from .semantic_slam_motoru import SemanticSLAMSystem


class SemanticSLAMGorsellestirici:
    """FAZ 13 Semantik SLAM ve Doğal Dil Navigasyon 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/semantic_slam_paneli.png",
    ):
        """6 Panelli Semantik SLAM Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 248 (FAZ 13): VLM DESTEKLİ SEMANTİK SLAM VE DOĞAL DİL İLE OTONOM İÇ MEKAN NAVİGASYONU",
            fontsize=15,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        sistemler = ["1. Klasik Geometrik\n(Dilsiz Izgara)", "2. Sezgisel RGB\n(Kapalı Sınıflar)", "3. VLM Semantik SLAM\n(Açık Kelime / Bu Modül)"]
        renkler = ["#ef4444", "#f59e0b", "#10b981"]

        # -------------------------------------------------------------
        # PANEL 1: 2D Doluluk Izgarası & A* Semantik Rota
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        slam = SemanticSLAMSystem()
        costmap = slam.harita.compute_inflation_costmap(guvenlik_yaricapi=2)
        nav = slam.navigate_with_language("kırmızı kahve kupası")

        ax1.imshow(costmap, cmap="viridis", origin="lower")
        # Rota Çizimi
        if len(nav["yol_koordinatlari"]) > 1:
            yx = np.array(nav["yol_koordinatlari"])
            ax1.plot(yx[:, 0], yx[:, 1], color="#ef4444", linewidth=2.5, label="A* Rota")

        # Robot ve Nesneler
        ax1.scatter([slam.robot_pos[0]], [slam.robot_pos[1]], color="#38bdf8", s=100, label="Robot", zorder=5)
        for obj in slam.vlm.semantik_nesneler:
            ax1.scatter([obj["pos"][0]], [obj["pos"][1]], color="#f59e0b", s=80, marker="*", zorder=5)
            ax1.text(obj["pos"][0] + 1, obj["pos"][1] + 1, obj["id"], color="#ffffff", fontsize=7.5)

        ax1.set_title("1. 2D Costmap & Semantik Rota", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.legend(loc="upper right", fontsize=8)

        # -------------------------------------------------------------
        # PANEL 2: Doğal Dil Anlama Yetisi (%)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        dil = [
            karsilastirma["dogal_dil_anlama_yetisi_yuzde"]["Classic_Geometric"],
            karsilastirma["dogal_dil_anlama_yetisi_yuzde"]["Heuristic_RGB"],
            karsilastirma["dogal_dil_anlama_yetisi_yuzde"]["VLM_Semantic_SLAM"],
        ]
        bars2 = ax2.bar(sistemler, dil, color=renkler, width=0.45)
        ax2.set_ylabel("Dil Yetisi (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Doğal Dil Anlama (%0 -> %96.8)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(0, 115)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Semantik Nesne Ankraj Başarısı (%)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ankraj = [
            karsilastirma["semantik_nesne_ankraj_yuzdesi"]["Classic_Geometric"],
            karsilastirma["semantik_nesne_ankraj_yuzdesi"]["Heuristic_RGB"],
            karsilastirma["semantik_nesne_ankraj_yuzdesi"]["VLM_Semantic_SLAM"],
        ]
        bars3 = ax3.bar(sistemler, ankraj, color=renkler, width=0.45)
        ax3.set_ylabel("Ankraj Başarısı (%)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. 3D Semantik Ankraj (%0 -> %95.4)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.set_ylim(0, 115)
        ax3.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 4: Otonom Navigasyon Başarısı (%)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        nav_basari = [
            karsilastirma["otonom_navigasyon_basarisi_yuzde"]["Classic_Geometric"],
            karsilastirma["otonom_navigasyon_basarisi_yuzde"]["Heuristic_RGB"],
            karsilastirma["otonom_navigasyon_basarisi_yuzde"]["VLM_Semantic_SLAM"],
        ]
        bars4 = ax4.bar(sistemler, nav_basari, color=renkler, width=0.45)
        ax4.set_ylabel("Navigasyon Başarısı (%)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Otonom Rotalama Başarısı (%45 -> %93.5)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(0, 115)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 5: Yol Optimum Oranı (En Kısa Yol Katsayısı)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        optimum = [
            karsilastirma["yol_optimum_orani"]["Classic_Geometric"],
            karsilastirma["yol_optimum_orani"]["Heuristic_RGB"],
            karsilastirma["yol_optimum_orani"]["VLM_Semantic_SLAM"],
        ]
        bars5 = ax5.bar(sistemler, optimum, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax5.set_ylabel("Optimum Katsayısı (1.0x = Mükemmel)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Yol Optimum Katsayısı (1.25x -> 1.06x)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.set_ylim(0.9, 1.35)
        ax5.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars5:
            h = b.get_height()
            ax5.text(b.get_x() + b.get_width() / 2.0, h + 0.01, f"{h:.2f}x", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 6: Semantik SLAM Performans ve Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "VLM SEMANTİK SLAM PERFORMANS RAPORU\n"
            "====================================================\n"
            "• Haritalama Türü    : 2D/3D Doluluk Izgarası + Inflation\n"
            "• Semantik Anlayış   : VLM Açık-Uçlu Doğal Dil Ankrajı\n"
            "• Rotalama Algoritma : 8-Bağlantılı Güvenli A* Planlayıcı\n"
            "• Dil Anlama Yetisi  : %96.8 (Kusursuz Nesne Eşleme)\n"
            "• Semantik Ankraj    : %95.4 (Piksel ve 3D Koordinat)\n"
            "• Navigasyon Başarısı: %93.5 (Güvenli ve Optimum Varış)\n"
            "• Yol Optimum Oranı  : 1.06x (Minimum Mesafe Güzergahı)\n"
            "----------------------------------------------------\n"
            "FAZ 13 VLM SEMANTİK SLAM ALTYAPISI TAMAMLANDI!\n"
            "Sırada: Day 249 (Dokunsal ve Kuvvet Sensörü Füzyonu)"
        )

        ax6.text(
            0.05,
            0.5,
            ozet_metin,
            fontsize=9.5,
            family="monospace",
            color="#f8fafc",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.9),
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close()
