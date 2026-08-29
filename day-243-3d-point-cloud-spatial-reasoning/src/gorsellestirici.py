"""
PointNet++ 6 Panelli Görselleştirici Modülü (FAZ 13) (Day 243).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np
from .point_cloud_motoru import ornek_3d_fincan_bulutu_olustur


class PointCloudGorsellestirici:
    """FAZ 13 3D Spatial AI 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/point_cloud_paneli.png",
    ):
        """6 Panelli PointNet++ 3D Nokta Bulutu Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(
            "DAY 243 (FAZ 13): 3D NOKTA BULUTU VE MEKANSAL AKIL YÜRÜTME (SPATIAL AI - POINTNET++)",
            fontsize=16,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        modeller = ["1. 2D Depth CNN\n(Projeksiyon)", "2. Vanilla PointNet\n(Küresel Havuz)", "3. PointNet++\n(Hiyerarşik Küme)"]
        renkler = ["#ef4444", "#f59e0b", "#10b981"]

        # -------------------------------------------------------------
        # PANEL 1: Set Abstraction Hiyerarşi Akışı
        # -------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        katmanlar = ["1. Ham Nokta Bulutu (N=512)", "2. En Uzak Nokta (FPS N'=128)", "3. Küresel Komşuluk (Ball Query r=0.2)", "4. Yerel PointNet MLP + Max-Pool", "5. 3D Tutma Afordansı (Skor)"]
        degerler = [1.0, 1.5, 2.0, 2.7, 3.4]
        ax1.barh(katmanlar[::-1], degerler[::-1], color=["#38bdf8", "#8b5cf6", "#f59e0b", "#10b981", "#ec4899"], height=0.45)
        ax1.set_xlabel("Set Abstraction Katman Sırası", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. PointNet++ Hiyerarşik İşlem Hattı", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: Mekansal Segmentasyon / Tutma mIoU (%)
        # -------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        miou = [
            karsilastirma["mekansal_segmentasyon_miou"]["2D_Depth_CNN"],
            karsilastirma["mekansal_segmentasyon_miou"]["Vanilla_PointNet"],
            karsilastirma["mekansal_segmentasyon_miou"]["PointNetPlusPlus"],
        ]
        bars2 = ax2.bar(modeller, miou, color=renkler, width=0.45)
        ax2.set_ylabel("mIoU Başarı Oranı (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Mekansal Segmentasyon (%52 -> %88.2)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(0, 110)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Geometrik Tutma Başarısı (%)
        # -------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        tutma = [
            karsilastirma["geometrik_tutma_basarisi"]["2D_Depth_CNN"],
            karsilastirma["geometrik_tutma_basarisi"]["Vanilla_PointNet"],
            karsilastirma["geometrik_tutma_basarisi"]["PointNetPlusPlus"],
        ]
        bars3 = ax3.bar(modeller, tutma, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax3.set_ylabel("Grasp Başarı Oranı (%)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Geometrik Tutma Başarısı (%46.5 -> %93.5)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.set_ylim(0, 115)
        ax3.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 4: Yoğunluk Değişimine Dayanıklılık (%)
        # -------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        dayaniklilik = [
            karsilastirma["yogunluk_degisimine_dayaniklilik"]["2D_Depth_CNN"],
            karsilastirma["yogunluk_degisimine_dayaniklilik"]["Vanilla_PointNet"],
            karsilastirma["yogunluk_degisimine_dayaniklilik"]["PointNetPlusPlus"],
        ]
        bars4 = ax4.bar(modeller, dayaniklilik, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax4.set_ylabel("Dayanıklılık Skoru (%)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Yoğunluk Değişimi Dayanıklılığı (%30 -> %91)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(0, 115)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 5: 3D Fincan Nokta Bulutu ve Tutma Yüzeyi Projeksiyonu
        # -------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5, projection="3d")
        bulut = ornek_3d_fincan_bulutu_olustur(nokta_sayisi=512)
        renk_skorlari = np.where(bulut[:, 0] > 0.25, 0.95, 0.35)  # Kulp kısmı yüksek afordanslı

        scatter = ax5.scatter(
            bulut[:, 0], bulut[:, 1], bulut[:, 2],
            c=renk_skorlari, cmap="coolwarm", s=15, alpha=0.85
        )
        ax5.set_xlabel("X (m)", fontsize=9, color="#cbd5e1")
        ax5.set_ylabel("Y (m)", fontsize=9, color="#cbd5e1")
        ax5.set_zlabel("Z (m)", fontsize=9, color="#cbd5e1")
        ax5.set_title("5. 3D Fincan Tutma Afordansı (Kırmızı=Yüksek)", fontsize=11, color="#38bdf8", fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 6: PointNet++ Performans ve Özet Kartı
        # -------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis("off")

        ozet_metin = (
            "POINTNET++ 3D SPATIAL AI RAPORU\n"
            "====================================================\n"
            "• Mimari Modeli       : Hiyerarşik Set Abstraction (SA)\n"
            "• Nokta Örnekleme     : Farthest Point Sampling (FPS)\n"
            "• Yerel Gruplama      : Ball Query (r=0.2, K=16)\n"
            "• Grasp Başarısı      : %46.5 -> %93.5 (+%47.0 Artış)\n"
            "• Segmentasyon mIoU   : %88.2 (Lidar / RGB-D Hassasiyeti)\n"
            "• Yoğunluk Direnci    : %91.0 (Seyrek Noktalarda Kararlı)\n"
            "• Çıkarım Gecikmesi   : 16.2ms (~60 Hz Lidar Uyumlu)\n"
            "----------------------------------------------------\n"
            "FAZ 13 3D MEKANSAL ZEKA MODÜLÜ TAMAMLANDI!\n"
            "Sırada: Day 244 (3D Bounding Box & 6-DoF Pose Estimation)"
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
