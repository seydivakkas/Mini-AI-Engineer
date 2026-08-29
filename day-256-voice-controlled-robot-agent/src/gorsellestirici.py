"""
Ses Komutlu Robot Ajanı 6 Panelli Görselleştirici Modülü (FAZ 13) (Day 256).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class VoiceRobotGorsellestirici:
    """FAZ 13 Ses Komutlu Robot Ajanı Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/voice_robot_paneli.png",
    ):
        """6 Panelli Voice-Controlled Robotic Agent Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 256 (FAZ 13): SES KOMUTLU ROBOT AJANI (WHISPER + VLM + VLA İLE UÇTAN UCA SESLİ ROBOT İDARESİ)",
            fontsize=15,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        kontrolculer = ["1. Hardcoded\n(Anahtar Kelime)", "2. Salt Metin LLM\n(Görselsiz)", "3. Whisper+VLM+VLA\n(Bu Modül)"]
        renkler = ["#ef4444", "#f59e0b", "#10b981"]

        # -------------------------------------------------------------
        # PANEL 1: 3D Mekansal Temellendirme ve Yörünge Haritası
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        # Masa Alanı
        masa = patches.Rectangle((0.1, -0.4), 0.8, 0.8, linewidth=1.5, edgecolor="#64748b", facecolor="#1e293b", alpha=0.5, label="Çalışma Masası")
        ax1.add_patch(masa)

        # Nesne 1: Kırmızı Kupa
        kupa = patches.Circle((0.45, 0.15), 0.06, color="#ef4444", label="Kırmızı Kupa [0.45, 0.15]")
        ax1.add_patch(kupa)

        # Nesne 2: Su Isıtıcısı
        isiticisi = patches.Rectangle((0.65, -0.25), 0.12, 0.14, color="#38bdf8", label="Su Isıtıcısı [0.70, -0.18]")
        ax1.add_patch(isiticisi)

        # VLA Taşıma Yörünge Yayı
        t_arc = np.linspace(0, 1, 20)
        x_arc = 0.45 + (0.70 - 0.45) * t_arc
        y_arc = 0.15 + (-0.18 - 0.15) * t_arc + 0.15 * np.sin(np.pi * t_arc)
        ax1.plot(x_arc, y_arc, color="#10b981", linestyle="--", linewidth=2.5, marker="o", markersize=4, label="VLA Taşıma Yörüngesi")

        ax1.set_xlim(0.0, 1.0)
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xlabel("X Koordinatı (m)", fontsize=9, color="#cbd5e1")
        ax1.set_ylabel("Y Koordinatı (m)", fontsize=9, color="#cbd5e1")
        ax1.set_title("1. VLM 3D Temellendirme ve VLA Yörünge İcrası", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: Doğal Ses Komut Anlama Oranı (%)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        anlama = [
            karsilastirma["dogal_ses_komut_anlama_yuzde"]["Hardcoded_Keyword"],
            karsilastirma["dogal_ses_komut_anlama_yuzde"]["Text_Only_LLM"],
            karsilastirma["dogal_ses_komut_anlama_yuzde"]["Whisper_VLM_VLA"],
        ]
        bars2 = ax2.bar(kontrolculer, anlama, color=renkler, width=0.45)
        ax2.set_ylabel("Komut Anlama Başarısı (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Doğal Ses Komut Anlama (%42 -> %98.4)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(0, 115)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Mekansal Temellendirme Doğruluğu (%)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        grounding = [
            karsilastirma["mekansal_temellendirme_dogrulugu_yuzde"]["Hardcoded_Keyword"],
            karsilastirma["mekansal_temellendirme_dogrulugu_yuzde"]["Text_Only_LLM"],
            karsilastirma["mekansal_temellendirme_dogrulugu_yuzde"]["Whisper_VLM_VLA"],
        ]
        bars3 = ax3.bar(kontrolculer, grounding, color=renkler, width=0.45)
        ax3.set_ylabel("Temellendirme Doğruluğu (%)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Mekansal Temellendirme (Grounding) (%35 -> %97.2)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.set_ylim(0, 115)
        ax3.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 4: Belirsizlik Çözümleme Başarısı (%)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        cozumleme = [
            karsilastirma["belirsizlik_cozumleme_basarisi_yuzde"]["Hardcoded_Keyword"],
            karsilastirma["belirsizlik_cozumleme_basarisi_yuzde"]["Text_Only_LLM"],
            karsilastirma["belirsizlik_cozumleme_basarisi_yuzde"]["Whisper_VLM_VLA"],
        ]
        bars4 = ax4.bar(kontrolculer, cozumleme, color=renkler, width=0.45)
        ax4.set_ylabel("Çözümleme Başarısı (%)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Belirsizlik ve Netleştirme Başarısı (%20 -> %96.5)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(0, 115)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 5: Uçtan Uca Tepki Gecikmesi (Milisaniye)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        gecikme = [
            karsilastirma["uctan_uca_tepki_gecikmesi_ms"]["Hardcoded_Keyword"],
            karsilastirma["uctan_uca_tepki_gecikmesi_ms"]["Text_Only_LLM"],
            karsilastirma["uctan_uca_tepki_gecikmesi_ms"]["Whisper_VLM_VLA"],
        ]
        bars5 = ax5.bar(kontrolculer, gecikme, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax5.set_ylabel("Gecikme (ms - Düşük İyi)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Uçtan Uca Tepki Gecikmesi (1400ms -> 220ms)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.set_ylim(0, 1600)
        ax5.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars5:
            h = b.get_height()
            ax5.text(b.get_x() + b.get_width() / 2.0, h + 30.0, f"{int(h)} ms", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 6: Voice Robot Performans ve Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "SES KOMUTLU ROBOT AJANI RAPORU\n"
            "====================================================\n"
            "• Konuşma Tanıma (ASR): OpenAI Whisper (16 kHz Mel)\n"
            "• Semantik Planlayıcı : LLM Task Decomposition\n"
            "• Görsel Temellendirme: VLM 3D Bounding Box Grounding\n"
            "• Eylem İcrası        : VLA Policy (v <= 0.25 m/s)\n"
            "• Komut Anlama        : %98.4 (Zirve Doğallık)\n"
            "• 3D Temellendirme    : %97.2 (Hassas Koordinat Eşleme)\n"
            "• Netleştirme Diyaloğu: %96.5 (Belirsizlikte Soru Sorma)\n"
            "• Tepki Süresi        : 220 ms (Gerçek Zamanlı Akış)\n"
            "----------------------------------------------------\n"
            "FAZ 13 SES KOMUTLU ROBOT AJANI TAMAMLANDI!\n"
            "Sırada: Day 257 (Robotic Safety & Force Limiting ISO 15066)"
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
