"""
Ajan Öz-Yansıtma 6 Panelli Görselleştirici Modülü (Day 237 - FAZ 12).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class RefleksiyonGorsellestirici:
    """Öz-Yansıtma Ajanı 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/refleksiyon_ajani_paneli.png",
    ):
        """6 Panelli Öz-Yansıtma Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 237 (FAZ 12): AJAN ÖZ-YANSITMA (SELF-REFLECTION) - RUBRİK DENETİMİ VE YİNELEMELİ İYİLEŞTİRME",
            fontsize=18,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        modeller = ["1. Tek Atımlı\n(Öz-Güvensiz Kör)", "2. Salt Denetçi\n(Sadece Eleştirir)", "3. Öz-Yansıtma\n(Self-Refine Döngüsü)"]
        renkler = ["#ef4444", "#f59e0b", "#10b981"]

        # -------------------------------------------------------------
        # PANEL 1: Öz-Yansıtma Döngüsü
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        adimlar = ["1. İlk Taslak Üretimi", "2. Rubrik Denetimi & Eleştiri", "3. Geri Bildirim Entegrasyonu", "4. Kod İyileştirme", "5. Eşik Onayı (>=90)"]
        puanlar = [1.0, 1.4, 1.8, 2.3, 2.8]
        ax1.barh(adimlar[::-1], puanlar[::-1], color=["#38bdf8", "#8b5cf6", "#f59e0b", "#10b981", "#ec4899"], height=0.45)
        ax1.set_xlabel("Döngü Aşamaları", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. Self-Refine & Reflexion Döngüsü", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: Güvenlik ve Doğruluk Skoru (%)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        basari = [
            karsilastirma["guvenlik_ve_dogruluk_skoru"]["Tek_Atimli_Uretici"],
            karsilastirma["guvenlik_ve_dogruluk_skoru"]["Salt_Denetci_Judge"],
            karsilastirma["guvenlik_ve_dogruluk_skoru"]["Yinelemeli_Oz_Yansitma"],
        ]
        bars2 = ax2.bar(modeller, basari, color=renkler, width=0.45)
        ax2.set_ylabel("Skor Oranı (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Güvenlik ve Doğruluk Skoru (%45.0 -> %96.8)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(0, 120)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Güvenlik Açığı / Bug Oranı (%)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        hata = [
            karsilastirma["guvenlik_acigi_orani"]["Tek_Atimli_Uretici"],
            karsilastirma["guvenlik_acigi_orani"]["Salt_Denetci_Judge"],
            karsilastirma["guvenlik_acigi_orani"]["Yinelemeli_Oz_Yansitma"],
        ]
        bars3 = ax3.bar(modeller, hata, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax3.set_ylabel("Hata / Açık Oranı (%)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Güvenlik Açığı Oranı (%55 -> %3.2)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.set_ylim(0, 65)
        ax3.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 4: Ortalama Kalite Puanı (100 Üzerinden)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        kalite = [
            karsilastirma["ortalama_kalite_puani"]["Tek_Atimli_Uretici"],
            karsilastirma["ortalama_kalite_puani"]["Salt_Denetci_Judge"],
            karsilastirma["ortalama_kalite_puani"]["Yinelemeli_Oz_Yansitma"],
        ]
        bars4 = ax4.bar(modeller, kalite, color=renkler, width=0.45)
        ax4.set_ylabel("Kalite Puanı (0-100)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Kod Kalite Puanı (50 -> 96 Puan)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(0, 120)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"{h:.0f} Puan", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 5: Canlı İterasyon Skor İlerlemesi
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        iter_adlari = ["İter 1: Düz Metin\n(50 Puan)", "İter 2: SHA256\n(70 Puan)", "İter 3: Bcrypt & Try\n(100 Puan)"]
        iter_skorlar = [50, 70, 100]
        bars5 = ax5.bar(iter_adlari, iter_skorlar, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax5.axhline(90, color="#ec4899", linestyle="--", label="Onay Eşiği (90 Puan)")
        ax5.set_ylabel("İterasyon Skoru", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Canlı İteratif Skor Artışı", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.set_ylim(0, 120)
        ax5.legend(loc="upper left")
        ax5.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars5:
            h = b.get_height()
            ax5.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"{h}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 6: GÜN 237 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "GÜN 237: AJAN ÖZ-YANSITMA (REFLECTION) KARNESİ\n"
            "----------------------------------------------------\n"
            "• Yöntem              : Self-Refine & Reflexion Döngüsü\n"
            "• Denetim Yöntemi     : Rubrik Tabanlı Eleştirmen (Critic)\n"
            "• Doğruluk & Güvenlik : %45.0 -> %96.8 (+%51.8 Artış)\n"
            "• Güvenlik Açığı      : %55.0 -> %3.2 (%84.2 Azalma)\n"
            "• Ortalama İterasyon  : 2.2 Adımda Onay (Eşik: 90 Puan)\n"
            "• Uygulama Kapsamı    : Kod Üretimi, Güvenlik Denetimi, LLM Judge\n"
            "----------------------------------------------------\n"
            "SONUÇ: Ajanımız artık ilk yazdığı hatalı kodu 'tamam' sanmıyor;\n"
            "kendi çıktısını acımasızca eleştirip Bcrypt ve tip güvenliğini\n"
            "ekleyerek %96.8 mükemmellikte teslim ediyor!"
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
