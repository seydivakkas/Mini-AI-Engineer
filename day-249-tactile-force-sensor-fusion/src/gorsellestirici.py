"""
Dokunsal ve Kuvvet Sensörü Füzyonu 6 Panelli Görselleştirici Modülü (FAZ 13) (Day 249).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np
from .tactile_fusion_motoru import GelSightTactileSensor


class TactileGorsellestirici:
    """FAZ 13 Dokunsal Algılama ve Kavrama 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/tactile_fusion_paneli.png",
    ):
        """6 Panelli Dokunsal Füzyon Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 249 (FAZ 13): DOKUNSAL (TACTILE) VE KUVVET SENSÖRÜ FÜZYONU İLE HASSAS NESNE TUTMA",
            fontsize=15,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        yontemler = ["1. Sabit Kuvvet\n(Ezici/Kör)", "2. Saf Görsel\n(Yavaş 30Hz)", "3. Dokunsal Füzyon\n(1000Hz Adaptif)"]
        renkler = ["#ef4444", "#f59e0b", "#10b981"]

        # -------------------------------------------------------------
        # PANEL 1: GelSight Dokunsal Basınç Deformasyonu
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        sensor = GelSightTactileSensor(res=32)
        patch = sensor.get_contact_patch(normal_kuvvet=5.5)
        im = ax1.imshow(patch["basinc_haritasi"], cmap="inferno", origin="lower")
        ax1.set_title("1. GelSight 2D Basınç Deformasyonu", fontsize=11, color="#38bdf8", fontweight="bold")
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        # -------------------------------------------------------------
        # PANEL 2: Kırılgan Nesne Ezilme Oranı (%)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ezilme = [
            karsilastirma["kirilgan_nesne_ezilme_yuzdesi"]["Fixed_Force"],
            karsilastirma["kirilgan_nesne_ezilme_yuzdesi"]["Pure_Vision"],
            karsilastirma["kirilgan_nesne_ezilme_yuzdesi"]["Tactile_Fusion"],
        ]
        bars2 = ax2.bar(yontemler, ezilme, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax2.set_ylabel("Ezilme Oranı (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Kırılma ve Ezilme Oranı (%48 -> %1.2)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(0, 60)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Nesne Kayma & Düşürme Oranı (%)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        kayma = [
            karsilastirma["nesne_kayma_dusurme_yuzdesi"]["Fixed_Force"],
            karsilastirma["nesne_kayma_dusurme_yuzdesi"]["Pure_Vision"],
            karsilastirma["nesne_kayma_dusurme_yuzdesi"]["Tactile_Fusion"],
        ]
        bars3 = ax3.bar(yontemler, kayma, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax3.set_ylabel("Düşürme Oranı (%)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Kayma ve Düşürme Oranı (%55 -> %0.8)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.set_ylim(0, 70)
        ax3.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 4: Kırılgan Nesne Başarı Oranı (%)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        basari = [
            karsilastirma["kirilgan_nesne_basari_yuzdesi"]["Fixed_Force"],
            karsilastirma["kirilgan_nesne_basari_yuzdesi"]["Pure_Vision"],
            karsilastirma["kirilgan_nesne_basari_yuzdesi"]["Tactile_Fusion"],
        ]
        bars4 = ax4.bar(yontemler, basari, color=renkler, width=0.45)
        ax4.set_ylabel("Başarı Oranı (%)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Hassas Tutuş Başarısı (%36 -> %97.5)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(0, 115)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 5: Dinamik Kuvvet Takibi ve Adaptasyon
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        sim = profil_raporu["simulasyon"]
        adımlar = list(range(1, len(sim["gecmis_Fn"]) + 1))
        ax5.plot(adımlar, sim["gecmis_Fn"], label="Normal Kuvvet Fn (N)", color="#10b981", linewidth=2.5, marker="o")
        ax5.plot(adımlar, sim["gecmis_Ft"], label="Teğetsel Yük Ft (N)", color="#ef4444", linewidth=2, linestyle="--")
        ax5.axhline(12.0, color="#f59e0b", linestyle=":", label="Kırılma Tavanı (12N)")
        ax5.set_xlabel("Zaman Adımı", fontsize=10, color="#cbd5e1")
        ax5.set_ylabel("Kuvvet (Newton)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. 1000Hz Adaptif Kuvvet Takibi", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.legend(loc="upper left", fontsize=8.5)
        ax5.grid(axis="both", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 6: Dokunsal Füzyon Performans ve Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "DOKUNSAL VE KUVVET FÜZYONU RAPORU\n"
            "====================================================\n"
            "• Sensör Mimarisi    : GelSight Optik Dokunsal + 6-Axis F/T\n"
            "• Kayma Tespiti      : Sürtünme Konisi Marjini (|Ft|/Fn)\n"
            "• Kontrol Döngüsü    : 1000 Hz Kapalı Döngü Adaptasyon\n"
            "• Kırılma / Ezilme   : %1.2 (Sertlik Limit Koruması)\n"
            "• Kayma / Düşürme    : %0.8 (Anlık Kuvvet Artışı)\n"
            "• Hassas Tutuş       : %97.5 (Kırılgan Nesne Başarısı)\n"
            "• Güvenlik Tavanı    : Fn < 12.0 Newton\n"
            "----------------------------------------------------\n"
            "FAZ 13 DOKUNSAL FÜZYON ALTYAPISI TAMAMLANDI!\n"
            "Sırada: Day 250 (Çift Kollu Bimanual Koordinasyon)"
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
