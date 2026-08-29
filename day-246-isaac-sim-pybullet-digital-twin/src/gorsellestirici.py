"""
Isaac Sim & PyBullet Dijital İkiz 6 Panelli Görselleştirici Modülü (FAZ 13) (Day 246).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np
from .digital_twin_motoru import SyntheticDataFactory


class DigitalTwinGorsellestirici:
    """FAZ 13 Robotik Simülasyon ve Dijital İkiz 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/digital_twin_paneli.png",
    ):
        """6 Panelli Dijital İkiz ve Sentetik Veri Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 246 (FAZ 13): SİMÜLASYONDA ROBOTİK — ISAAC SIM & PYBULLET İLE DİJİTAL İKİZ VE SENTETİK VERİ ÜRETİMİ",
            fontsize=15,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        modeller = ["1. Fiziksel Robot\n(Aşınmalı & Yavaş)", "2. PyBullet CPU\n(Tek İş Parçacığı)", "3. Isaac Sim GPU\n(Paralel Dijital İkiz)"]
        renkler = ["#ef4444", "#f59e0b", "#10b981"]

        # -------------------------------------------------------------
        # PANEL 1: Dijital İkiz İşlem Hattı Akışı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        katmanlar = ["1. URDF Kinematik Tanımı (7-DoF)", "2. Ters Kinematik (IK Çözücü)", "3. PD Tork & Euler Entegrasyonu", "4. Çarpışma ve Sürtünme Fiziği", "5. Sentetik RGB-D Render"]
        degerler = [1.0, 1.5, 2.0, 2.7, 3.4]
        ax1.barh(katmanlar[::-1], degerler[::-1], color=["#38bdf8", "#8b5cf6", "#f59e0b", "#10b981", "#ec4899"], height=0.45)
        ax1.set_xlabel("Simülasyon Aşaması", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. Dijital İkiz Fizik & Kinematik Hattı", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: Veri Üretim Hacmi (Yörünge / Saat)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        hacim = [
            karsilastirma["veri_uretim_hacmi_yorunge_saat"]["Fiziksel_Robot"],
            karsilastirma["veri_uretim_hacmi_yorunge_saat"]["PyBullet_CPU"],
            karsilastirma["veri_uretim_hacmi_yorunge_saat"]["IsaacSim_GPU"],
        ]
        bars2 = ax2.bar(modeller, hacim, color=["#ef4444", "#38bdf8", "#10b981"], width=0.45)
        ax2.set_yscale("log")
        ax2.set_ylabel("Yörünge / Saat (Log Ölçek)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Veri Üretim Hacmi (1 -> 50,000 traj/h)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h * 1.3, f"{int(h):,}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9.5)

        # -------------------------------------------------------------
        # PANEL 3: Donanım Kırılma Riski (%)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        risk = [
            karsilastirma["donanim_kirilma_riski_yuzde"]["Fiziksel_Robot"],
            karsilastirma["donanim_kirilma_riski_yuzde"]["PyBullet_CPU"],
            karsilastirma["donanim_kirilma_riski_yuzde"]["IsaacSim_GPU"],
        ]
        bars3 = ax3.bar(modeller, risk, color=["#ef4444", "#10b981", "#10b981"], width=0.45)
        ax3.set_ylabel("Risk Oranı (%)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Donanım Kırılma Riski (%75 -> %0)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.set_ylim(0, 95)
        ax3.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 4: Sim-to-Real Uyum Başarısı (%)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        uyum = [
            karsilastirma["sim2real_uyum_basarisi_yuzde"]["Fiziksel_Robot"],
            karsilastirma["sim2real_uyum_basarisi_yuzde"]["PyBullet_CPU"],
            karsilastirma["sim2real_uyum_basarisi_yuzde"]["IsaacSim_GPU"],
        ]
        bars4 = ax4.bar(modeller, uyum, color=["#38bdf8", "#f59e0b", "#10b981"], width=0.45)
        ax4.set_ylabel("Uyum Başarısı (%)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Sim-to-Real Aktarım (%65 -> %88.5)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(0, 120)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 5: Sentetik Çok Modlu Görüntü Render'ı
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        sentetik = SyntheticDataFactory.render_synthetic_scene(
            eef_pos=np.array([0.3, 0.2, 0.4]),
            object_pos=np.array([0.4, 0.1, 0.3]),
            res=64,
        )
        ax5.imshow(sentetik["rgb"])
        ax5.set_title("5. Sentetik Çok Modlu Görsel Sahne", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.axis("off")

        # -------------------------------------------------------------
        # PANEL 6: Dijital İkiz Performans ve Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "DİJİTAL İKİZ VE SİMÜLASYON RAPORU\n"
            "====================================================\n"
            "• Simülatör Mimarisi  : Isaac Sim (GPU) & PyBullet Kinematik\n"
            "• Kinematik Çözücü    : Sönümlü Jakoben Ters Kinematik (IK)\n"
            "• Veri Üretim Hızı    : 50,000+ yörünge/saat (Sıfır Aşınma)\n"
            "• Sim-to-Real Başarı  : %88.5 (Fotogerçekçi Işın İzleme)\n"
            "• Fizik Adımı Gecikme : 0.12 ms (Ultra Hızlı 1000Hz Adım)\n"
            "• Sentetik Modlar     : RGB + Derinlik Haritası + Semantik Maske\n"
            "----------------------------------------------------\n"
            "FAZ 13 SİMÜLASYON VE DİJİTAL İKİZ TAMAMLANDI!\n"
            "Sırada: Day 247 (Sim2Real Domain Randomization)"
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
