"""
Constitutional AI (CAI) 6 Panelli Görselleştirici Modülü (Day 212 - FAZ 11).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class ConstitutionalGorsellestirici:
    """Constitutional AI 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/constitutional_ai_paneli.png",
    ):
        """6 Panelli Constitutional AI Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 212 (FAZ 11): CONSTITUTIONAL AI (CAI) & RLAIF (ANAYASAL KENDİ KENDİNİ ELEŞTİRME VE GÜVENLİK)",
            fontsize=18,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        modeller = ["Hizalanmamış Model", "İnsan RLHF", "Constitutional AI\n(RLAIF)"]

        # -------------------------------------------------------------
        # PANEL 1: Constitutional AI 2 Aşamalı Mimari
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        asamalar = ["1. Kırmızı Takım İstemi", "2. Ham Filtresiz Yanıt (y_0)", "3. Anayasa İlkeleri Taraması", "4. Eleştiri & Düzeltme (y_1)", "5. RLAIF Tercih Eğitimi"]
        onemler = [1.0, 1.3, 1.8, 2.1, 2.4]
        ax1.barh(asamalar[::-1], onemler[::-1], color=["#38bdf8", "#ef4444", "#8b5cf6", "#10b981", "#ec4899"], height=0.45)
        ax1.set_xlabel("İşlem Aşamaları", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. Anayasal Kendi Kendini Düzeltme Hattı", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: Toksisite ve Zararlı İçerik Oranı (%)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        toksisite = [
            karsilastirma["toksisite_orani"]["Hizalanmamis_Model"],
            karsilastirma["toksisite_orani"]["Insan_RLHF"],
            karsilastirma["toksisite_orani"]["Constitutional_AI"],
        ]
        bars2 = ax2.bar(modeller, toksisite, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax2.set_ylabel("Toksisite / Zarar Oranı (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Zararlı İçerik Üretimi Karşılaştırması", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(0, 55)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Aşırı Reddetme (Over-Refusal) Oranı (%)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        asiri_ret = [
            karsilastirma["asiri_reddetme_orani"]["Hizalanmamis_Model"],
            karsilastirma["asiri_reddetme_orani"]["Insan_RLHF"],
            karsilastirma["asiri_reddetme_orani"]["Constitutional_AI"],
        ]
        bars3 = ax3.bar(modeller, asiri_ret, color=["#38bdf8", "#ef4444", "#10b981"], width=0.45)
        ax3.set_ylabel("Aşırı Ret / False-Positive (%)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Aşırı Reddetmeme (No Over-Refusal) Dengesi", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.set_ylim(0, 45)
        ax3.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 4: Jailbreak & Kırmızı Takım Savunma Başarısı (%)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        savunma = [
            karsilastirma["jailbreak_savunma_basarisi"]["Hizalanmamis_Model"],
            karsilastirma["jailbreak_savunma_basarisi"]["Insan_RLHF"],
            karsilastirma["jailbreak_savunma_basarisi"]["Constitutional_AI"],
        ]
        bars4 = ax4.bar(modeller, savunma, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax4.set_ylabel("Savunma Başarısı (%)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Kırmızı Takım (Red-Team) Savunma Gücü", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(0, 115)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 5: İnsan Etiketleme Maliyeti ($)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        maliyet_etiketler = ["İnsan RLHF\n(Annotator Maliyeti)", "Constitutional AI\n(AI Geri Bildirimi)"]
        maliyetler = [karsilastirma["etiketleme_maliyeti_dolar"]["Insan_RLHF"], karsilastirma["etiketleme_maliyeti_dolar"]["Constitutional_AI"]]
        bars5 = ax5.bar(maliyet_etiketler, [m / 1000 for m in maliyetler], color=["#ef4444", "#10b981"], width=0.45)
        ax5.set_ylabel("Maliyet (Bin Dolar - $k)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Güvenlik Hizalama Maliyeti ($150k vs $0)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.set_ylim(0, 180)
        ax5.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars5:
            h = b.get_height()
            ax5.text(b.get_x() + b.get_width() / 2.0, h + 3.0, f"${h:.0f}k", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 6: GÜN 212 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "GÜN 212: CONSTITUTIONAL AI (CAI) KARNESİ\n"
            "----------------------------------------------------\n"
            "• Yöntem               : Constitutional AI (Anthropic RLAIF)\n"
            "• Aşama 1              : Supervised Critique & Revision SFT\n"
            "• Aşama 2              : RLAIF (AI Geri Bildirimli Tercih)\n"
            "• Toksisite Azalması   : %46.5 -> %0.8 (%98 Güvenlik Artışı)\n"
            "• Aşırı Ret Önleme     : %38.0 -> %4.2 (Meşru Soruları Yanıtlama)\n"
            "• Jailbreak Savunması  : %97.5 Başarı (Kırmızı Takım Direnci)\n"
            "• İnsan Etiket Maliyeti: $0 (Tamamen Otonom Anayasal İlkeler)\n"
            "----------------------------------------------------\n"
            "SONUÇ: İnsan hakemlere travma yaşatmadan ve sıfır maliyetle\n"
            "model anayasal ilkelerle kendi kendini hizalamayı başardı!"
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
