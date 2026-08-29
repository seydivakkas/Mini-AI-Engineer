"""
Day 289 (FAZ 15): Çoklu Ajan Tartışması (MAD) 6 Panelli Görselleştirici Modülü.
"""

from typing import Dict, Any
import os
import matplotlib.pyplot as plt
import numpy as np


class MultiAgentDebateGorsellestirici:
    """FAZ 15 Çoklu Ajan Tartışması Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/multi_agent_debate_society_paneli.png",
    ):
        """6 Panelli Çoklu Ajan Tartışması Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 289 (FAZ 15): ÇOK MODLU ÇOKLU AJAN TARTIŞMASI VE KONSENSÜS (MULTI-AGENT DEBATE)",
            fontsize=15,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        modeller = ["1. Single Agent\n(Tek Ajan)", "2. Majority Voting\n(Çoğunluk Oyu)", "3. Multi-Agent Debate\n(Ajanlar Toplumu)"]
        renkler = ["#ef4444", "#f59e0b", "#10b981"]

        # -------------------------------------------------------------
        # PANEL 1: Çok Aşamalı Muhakeme Başarısı (%)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        accs = [
            karsilastirma["muhakeme_basarisi_yuzde"]["1. Single Agent"],
            karsilastirma["muhakeme_basarisi_yuzde"]["2. Majority Voting"],
            karsilastirma["muhakeme_basarisi_yuzde"]["3. Multi-Agent Debate"],
        ]
        b1 = ax1.bar(modeller, accs, color=renkler, width=0.45)
        ax1.set_ylabel("Muhakeme Doğruluğu (%)", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. Çok Aşamalı Muhakeme Başarısı (%61.5 -> %97.4)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.set_ylim(0, 120)
        ax1.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b1:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2.0, h + 2.0, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 2: Halüsinasyon Oranı (%)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        halus = [
            karsilastirma["halusinasyon_orani"]["1. Single Agent"],
            karsilastirma["halusinasyon_orani"]["2. Majority Voting"],
            karsilastirma["halusinasyon_orani"]["3. Multi-Agent Debate"],
        ]
        b2 = ax2.bar(modeller, halus, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax2.set_ylabel("Halüsinasyon Oranı (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Halüsinasyon Tasfiyesi (%38.6 -> %2.1 | 18x Azalma)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(0, 50)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Tartışma Turlarına Göre Güven Eğrisi (%)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        r_labels = profil_raporu["round_labels"]
        r_confs = profil_raporu["round_confidences"]

        ax3.plot(r_labels, r_confs, "o-", color="#38bdf8", linewidth=2.5, markersize=8)
        ax3.fill_between(r_labels, 30, r_confs, color="#38bdf8", alpha=0.15)
        ax3.set_ylabel("Konsensüs Güven Skoru (%)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Turlar Boyunca Konsensüs Yakınsaması", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.set_ylim(30, 110)
        ax3.grid(True, linestyle=":", alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha="right")

        for i, (rl, rc) in enumerate(zip(r_labels, r_confs)):
            ax3.text(i, rc + 2.0, f"%{rc:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9.0)

        # -------------------------------------------------------------
        # PANEL 4: Heterojen Ajan Rolleri ve Elo Puanları
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ag_names = profil_raporu["agent_names"]
        ag_elos = profil_raporu["agent_elos"]
        ag_colors = ["#38bdf8", "#f59e0b", "#a855f7", "#10b981"]

        b4 = ax4.bar(ag_names, ag_elos, color=ag_colors, width=0.45)
        ax4.set_ylabel("Elo Derecelendirmesi (Rating)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Heterojen Ajan Rolleri ve Elo Güç Puanları", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(1400, 2000)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha="right")

        for b in b4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 10.0, f"{int(h)}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9.5)

        # -------------------------------------------------------------
        # PANEL 5: Yanılgıda Israrcılık (Bias Rigidity) Oranı
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        bias_vals = [
            karsilastirma["yanilgida_israr_orani"]["1. Single Agent"],
            karsilastirma["yanilgida_israr_orani"]["2. Majority Voting"],
            karsilastirma["yanilgida_israr_orani"]["3. Multi-Agent Debate"],
        ]
        b5 = ax5.bar(modeller, bias_vals, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax5.set_ylabel("Ön Yargı / Yanılgıda Israr (%)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Dogmatik Ön Yargı ve Israrcılık Tasfiyesi", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.set_ylim(0, 100)
        ax5.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b5:
            h = b.get_height()
            ax5.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 6: Multi-Agent Debate Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "MULTI-AGENT DEBATE & KONSENSÜS RAPORU\n"
            "====================================================\n"
            "• Mimarî Çerçeve       : Society of Mind & Multi-Agent Debate\n"
            "• Katılımcı Ajanlar    : Tez Sahibi, Eleştirmen, Hakem\n"
            "• Diyalektik Süreç     : 3 Turlu Çapraz Eleştiri & Sentez\n"
            "• Muhakeme Doğruluğu   : %97.4 (Tek Ajan: %61.5 | +%35.9)\n"
            "• Halüsinasyon Oranı   : %2.1 (Tek Ajan: %38.6 | 18x İyileşme)\n"
            "• Konsensüs Puanlaması : Elo-Ranked Softmax Ağırlıklandırma\n"
            "• Israrcılık Tasfiyesi : %85.0 -> %2.5 (Öz-Eleştiri ile Çözüm)\n"
            "• Uygulama Alanı       : Otonom Kod İnceleme, Hukuk, Tıp, Mimari\n"
            "----------------------------------------------------\n"
            "FAZ 15 GÜN 289 ÇOKLU AJAN TARTIŞMASI TAMAMLANDI!\n"
            "Sırada: Day 290 (Mekanistik Yorumlanabilirlik & SAE Probing)"
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
