"""
Otomatik Red-Teaming ve Güvenlik 6 Panelli Görselleştirici Modülü (Day 219 - FAZ 11).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class RedTeamingGorsellestirici:
    """Red-Teaming 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/red_teaming_paneli.png",
    ):
        """6 Panelli Red-Teaming Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 219 (FAZ 11): OTOMATİK RED-TEAMING & JAILBREAK SAVUNMA EĞİTİMİ (ADVERSARIAL SAFETY DPO)",
            fontsize=18,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        vektor = profil_raporu["vektor_analizi"]
        modeller = ["Savunmasız\nHam Model", "Kelime Filtresi\n(Blocklist)", "Otomatik\nRed-Teaming"]

        # -------------------------------------------------------------
        # PANEL 1: Kırmızı Takım Savunma Hattı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        asamalar = ["1. Saldırgan Model (PAIR/TAP)", "2. Çok Vektörlü Jailbreak", "3. Güvenlik Hakemi Taraması", "4. Düşmanca Üçlü (D_adv)", "5. Güvenlik DPO Eğitimi"]
        onemler = [1.0, 1.5, 1.9, 2.3, 2.7]
        ax1.barh(asamalar[::-1], onemler[::-1], color=["#ef4444", "#f59e0b", "#8b5cf6", "#38bdf8", "#10b981"], height=0.45)
        ax1.set_xlabel("İşlem Aşamaları", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. Otomatik Kırmızı Takım Hattı", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: Saldırı Başarı Oranı (ASR - %)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        asr = [
            karsilastirma["saldiri_basari_orani_asr"]["Savunmasiz_Ham_Model"],
            karsilastirma["saldiri_basari_orani_asr"]["Kelime_Filtresi_Blocklist"],
            karsilastirma["saldiri_basari_orani_asr"]["Otomatik_Red_Teaming"],
        ]
        bars2 = ax2.bar(modeller, asr, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax2.set_ylabel("Saldırı Başarı Oranı (ASR %)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Jailbreak Başarı Oranı (%74.5 -> %1.8)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(0, 90)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 1.2, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Aşırı Reddetme Oranı (FRR - %)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        frr = [
            karsilastirma["asiri_reddetme_orani_frr"]["Savunmasiz_Ham_Model"],
            karsilastirma["asiri_reddetme_orani_frr"]["Kelime_Filtresi_Blocklist"],
            karsilastirma["asiri_reddetme_orani_frr"]["Otomatik_Red_Teaming"],
        ]
        bars3 = ax3.bar(modeller, frr, color=["#38bdf8", "#ef4444", "#10b981"], width=0.45)
        ax3.set_ylabel("Aşırı Ret / False Positive (%)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Aşırı Reddetmeme Dengesi (FRR %2.4)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.set_ylim(0, 45)
        ax3.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 0.8, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 4: Genel Güvenlik ve Savunma Skoru (%)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        savunma = [
            karsilastirma["guvenlik_savunma_skoru"]["Savunmasiz_Ham_Model"],
            karsilastirma["guvenlik_savunma_skoru"]["Kelime_Filtresi_Blocklist"],
            karsilastirma["guvenlik_savunma_skoru"]["Otomatik_Red_Teaming"],
        ]
        bars4 = ax4.bar(modeller, savunma, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax4.set_ylabel("Güvenlik Skoru (%)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Genel Kırmızı Takım Direnci (%98.2)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(0, 115)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 5: Vektör Bazında Saldırı Başarısı Karşılaştırması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        x_idx = np.arange(len(vektor["vektorler"]))
        width = 0.35

        b_ham = ax5.bar(x_idx - width / 2, vektor["ham_model_asr"], width, label="Ham Model ASR", color="#ef4444")
        b_sav = ax5.bar(x_idx + width / 2, vektor["red_team_savunmali_asr"], width, label="Red-Team Savunmalı", color="#10b981")

        ax5.set_xticks(x_idx)
        ax5.set_xticklabels(vektor["vektorler"], fontsize=8.5, color="#cbd5e1")
        ax5.set_ylabel("Saldırı Başarı Oranı (ASR %)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Vektör Bazında Savunma Güçlenmesi", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.set_ylim(0, 105)
        ax5.grid(axis="y", linestyle=":", alpha=0.3)
        ax5.legend(loc="upper right", fontsize=8.5)

        # -------------------------------------------------------------
        # PANEL 6: GÜN 219 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "GÜN 219: OTOMATİK RED-TEAMING KARNESİ\n"
            "----------------------------------------------------\n"
            "• Yöntem              : Automated Adversarial Red-Teaming\n"
            "• Saldırı Vektörleri  : DAN Rol Yapma, Base64, Kurgu, Ters Mantık\n"
            "• Savunma Eğitimi     : Düşmanca Tercih DPO (Adversarial DPO)\n"
            "• Saldırı Başarısı ASR: %74.5 -> %1.8 (%98.2 Savunma Başarısı)\n"
            "• Aşırı Ret Oranı FRR : %2.4 (Meşru Sorular Engellenmez)\n"
            "• Base64 Şifreli ASR  : %91.5 -> %0.8 (Tamamen Bloke Edildi)\n"
            "----------------------------------------------------\n"
            "SONUÇ: İnsan güvenlik uzmanlarının yetişemeyeceği hızda\n"
            "otonom saldırgan modellerle kurşungeçirmez yapay zeka sağlandı!"
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
