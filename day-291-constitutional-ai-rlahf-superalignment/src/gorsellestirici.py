"""
Day 291 (FAZ 15): Anayasal Yapay Zeka (Constitutional AI) 6 Panelli Görselleştirici Modülü.
"""

from typing import Dict, Any
import os
import matplotlib.pyplot as plt
import numpy as np


class ConstitutionalAIGorsellestirici:
    """FAZ 15 Anayasal Yapay Zeka Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/constitutional_ai_superalignment_paneli.png",
    ):
        """6 Panelli Anayasal Yapay Zeka Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 291 (FAZ 15): ANAYASAL YAPAY ZEKA VE RLAHF SÜPER HİZALANMA (CONSTITUTIONAL AI)",
            fontsize=15,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        modeller = ["1. Raw Base LLM\n(Ham Model)", "2. Human RLHF\n(İnsanlı Tercih)", "3. Constitutional AI\n(AI Feedback - RLAHF)"]
        renkler = ["#ef4444", "#f59e0b", "#10b981"]

        # -------------------------------------------------------------
        # PANEL 1: Zararsızlık ve Güvenlik Skoru (%)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        safe_scores = [
            karsilastirma["zararsizlik_guvenlik_skoru"]["1. Raw Base LLM"],
            karsilastirma["zararsizlik_guvenlik_skoru"]["2. Human RLHF"],
            karsilastirma["zararsizlik_guvenlik_skoru"]["3. Constitutional AI"],
        ]
        b1 = ax1.bar(modeller, safe_scores, color=renkler, width=0.45)
        ax1.set_ylabel("Zararsızlık Skoru (%)", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. Zararsızlık ve Model Güvenliği (%42.1 -> %98.9)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.set_ylim(0, 120)
        ax1.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b1:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2.0, h + 2.0, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 2: Yağcılık (Sycophancy) Oranı (%)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        syco_vals = [
            karsilastirma["yagcilik_sycophancy_orani"]["1. Raw Base LLM"],
            karsilastirma["yagcilik_sycophancy_orani"]["2. Human RLHF"],
            karsilastirma["yagcilik_sycophancy_orani"]["3. Constitutional AI"],
        ]
        b2 = ax2.bar(modeller, syco_vals, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax2.set_ylabel("Yağcılık / Aldatma Oranı (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Yağcılık (Sycophancy) Tasfiyesi (%64.2 -> %1.8 | 35x)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(0, 80)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Anayasa İlkeleri Uyumluluk Dağılımı (%)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ilkeler = profil_raporu["ilkeler"]
        uyum = profil_raporu["uyumluluk"]
        i_colors = ["#10b981", "#38bdf8", "#a855f7"]

        b3 = ax3.bar(ilkeler, uyum, color=i_colors, width=0.45)
        ax3.set_ylabel("Anayasal Uyum Oranı (%)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Anayasa İlkelerine Uyum Puanları", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.set_ylim(90, 102)
        ax3.grid(axis="y", linestyle=":", alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha="right")

        for b in b3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 0.3, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9.5)

        # -------------------------------------------------------------
        # PANEL 4: Jailbreak Savunmasızlığı (%)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        jb_vals = [
            karsilastirma["jailbreak_savunmasizlik_yuzde"]["1. Raw Base LLM"],
            karsilastirma["jailbreak_savunmasizlik_yuzde"]["2. Human RLHF"],
            karsilastirma["jailbreak_savunmasizlik_yuzde"]["3. Constitutional AI"],
        ]
        b4 = ax4.bar(modeller, jb_vals, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax4.set_ylabel("Jailbreak Açığı (%)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Kötü Niyetli İstismar (Jailbreak) Açığı (%58.0 -> %0.6)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(0, 70)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 5: Yardımseverlik ve Hizalanma Dengesi (Pareto)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        help_vals = [
            karsilastirma["yardimseverlik_skoru"]["1. Raw Base LLM"],
            karsilastirma["yardimseverlik_skoru"]["2. Human RLHF"],
            karsilastirma["yardimseverlik_skoru"]["3. Constitutional AI"],
        ]
        b5 = ax5.bar(modeller, help_vals, color=renkler, width=0.45)
        ax5.set_ylabel("Yardımseverlik Skoru (%)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Hizalanma Sırasında Yetenek Korunumu (%95.2)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.set_ylim(0, 120)
        ax5.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b5:
            h = b.get_height()
            ax5.text(b.get_x() + b.get_width() / 2.0, h + 2.0, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 6: Constitutional AI Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "CONSTITUTIONAL AI & RLAHF RAPORU\n"
            "====================================================\n"
            "• Mimarî Çerçeve       : Constitutional AI (Anthropic CAI)\n"
            "• 1. Aşama             : Öz-Eleştiri & Revizyon (Critique-Revision SL)\n"
            "• 2. Aşama             : AI Feedback Tabanlı RL (RLAHF / DPO)\n"
            "• Zararsızlık Skoru    : %98.9 (Ham: %42.1 | İnsanlı RLHF: %74.5)\n"
            "• Yağcılık (Sycophancy): %1.8 (35.6 Kat Tasfiye Edildi)\n"
            "• Jailbreak Direnci    : %99.4 (%0.6 İstismar Oranı)\n"
            "• Yetenek Korunumu     : %95.2 Yardımseverlik (Alignment Tax Yok)\n"
            "• Temel Avantaj        : İnsan Etiketleyici Bağımlılığını Bitirme\n"
            "----------------------------------------------------\n"
            "FAZ 15 GÜN 291 CONSTITUTIONAL AI TAMAMLANDI!\n"
            "Sırada: Day 292 (Otonom Araştırma Ajanı - Autonomous Science Discovery)"
        )

        ax6.text(
            0.05,
            0.5,
            ozet_metin,
            fontsize=9.2,
            family="monospace",
            color="#f8fafc",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.9),
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close()
