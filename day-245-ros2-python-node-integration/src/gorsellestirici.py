"""
ROS2 Python Entegrasyonu 6 Panelli Görselleştirici Modülü (FAZ 13) (Day 245).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class ROS2Gorsellestirici:
    """FAZ 13 ROS 2 Robotik İletişim 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/ros2_paneli.png",
    ):
        """6 Panelli ROS 2 Düğüm Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 245 (FAZ 13): ROS2 (ROBOT OPERATING SYSTEM) PYTHON ENTEGRASYONU VE SENSÖR-EYLEYİCİ İLETİŞİMİ",
            fontsize=16,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        protokoller = ["1. HTTP REST\n(Aşırı Başlık Yükü)", "2. Raw Sockets\n(Tipi Belirsiz)", "3. ROS 2 DDS\n(Sıfır Kopya IPC)"]
        renkler = ["#ef4444", "#f59e0b", "#10b981"]

        # -------------------------------------------------------------
        # PANEL 1: ROS 2 Düğüm & Konu Grafı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        dugumler = ["Camera Node (/camera/rgb)", "AI VLA Node (Inference)", "Grasp Service (/arm/grasp)", "Arm Controller (/joint_cmd)", "Hardware DDS Bus"]
        puanlar = [1.0, 1.6, 2.1, 2.7, 3.3]
        ax1.barh(dugumler[::-1], puanlar[::-1], color=["#38bdf8", "#8b5cf6", "#f59e0b", "#10b981", "#ec4899"], height=0.45)
        ax1.set_xlabel("İletişim Katmanları", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. ROS 2 Düğüm ve Konu Grafı", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: Mesaj İletim Gecikmesi (ms)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        gecikme = [
            karsilastirma["mesaj_iletilme_gecikmesi_ms"]["HTTP_REST"],
            karsilastirma["mesaj_iletilme_gecikmesi_ms"]["Raw_Sockets"],
            karsilastirma["mesaj_iletilme_gecikmesi_ms"]["ROS2_DDS"],
        ]
        bars2 = ax2.bar(protokoller, gecikme, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax2.set_ylabel("Gecikme Süresi (ms - Düşük İyi)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. İletişim Gecikmesi (45ms -> 0.42ms)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(0, 55)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"{h:.2f}ms", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Paket Jitter / Kayıp Oranı (%)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        kayip = [
            karsilastirma["paket_jitter_kaybi_yuzdesi"]["HTTP_REST"],
            karsilastirma["paket_jitter_kaybi_yuzdesi"]["Raw_Sockets"],
            karsilastirma["paket_jitter_kaybi_yuzdesi"]["ROS2_DDS"],
        ]
        bars3 = ax3.bar(protokoller, kayip, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax3.set_ylabel("Kayıp / Jitter (%)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Paket Kaybı ve Jitter (%12 -> %0.001)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.set_ylim(0, 15)
        ax3.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 0.3, f"%{h:.3f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 4: Maksimum Mesaj Hacmi (msg/sn)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        hacim = [
            karsilastirma["maksimum_mesaj_hacmi_msg_sn"]["HTTP_REST"],
            karsilastirma["maksimum_mesaj_hacmi_msg_sn"]["Raw_Sockets"],
            karsilastirma["maksimum_mesaj_hacmi_msg_sn"]["ROS2_DDS"],
        ]
        bars4 = ax4.bar(protokoller, hacim, color=["#ef4444", "#38bdf8", "#10b981"], width=0.45)
        ax4.set_ylabel("Mesaj Hacmi (msg/sn - Yüksek İyi)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Mesaj Hacmi (220 -> 10,000+ msg/s)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(0, 12000)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 200, f"{h:,}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9.5)

        # -------------------------------------------------------------
        # PANEL 5: Donanım Senkronizasyon Skoru (%)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        senkron = [
            karsilastirma["donanim_senkronizasyon_skoru"]["HTTP_REST"],
            karsilastirma["donanim_senkronizasyon_skoru"]["Raw_Sockets"],
            karsilastirma["donanim_senkronizasyon_skoru"]["ROS2_DDS"],
        ]
        bars5 = ax5.bar(protokoller, senkron, color=renkler, width=0.45)
        ax5.set_ylabel("Senkronizasyon (%)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Motor Senkronizasyonu (%35 -> %98.5)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.set_ylim(0, 120)
        ax5.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars5:
            h = b.get_height()
            ax5.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 6: ROS 2 Performans ve Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "ROS 2 PYTHON ENTEGRASYON RAPORU\n"
            "====================================================\n"
            "• İletişim Katmanı    : ROS 2 rclpy + DDS Middleware\n"
            "• Konu (Topics)       : /camera/rgb, /arm/joint_commands\n"
            "• Servis (RPC)        : /arm/grasp_planner\n"
            "• Gecikme             : 0.42 ms (Ultra Hızlı IPC İletimi)\n"
            "• Paket Kaybı         : %0.001 (Kusursuz Kararlılık)\n"
            "• Mesaj Kapasitesi    : 10,000+ msg/sn (Yüksek Bant)\n"
            "• Senkronizasyon      : %98.5 (Mikrosaniye Motor Kilidi)\n"
            "----------------------------------------------------\n"
            "FAZ 13 ROBOTİK İLETİŞİM ALTYAPISI TAMAMLANDI!\n"
            "Sırada: Day 246 (Isaac Sim & PyBullet Digital Twin)"
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
