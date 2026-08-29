"""
Day 292 (FAZ 15): Otonom Bilimsel Keşif (AI Scientist) 6 Panelli Görselleştirici Modülü.
"""

from typing import Dict, Any
import os
import matplotlib.pyplot as plt
import numpy as np


class AIScientistGorsellestirici:
    """FAZ 15 AI Scientist Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/ai_scientist_autonomous_discovery_paneli.png",
    ):
        """6 Panelli AI Scientist Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 292 (FAZ 15): OTONOM BİLİMSEL KEŞİF VE AI SCIENTIST (AUTONOMOUS DISCOVERY)",
            fontsize=15,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        modeller = ["1. Human Scientist\n(Geleneksel)", "2. Semi-Automated\n(Asistan Destekli)", "3. The AI Scientist\n(Tam Otonom)"]
        renkler = ["#ef4444", "#f59e0b", "#10b981"]

        # -------------------------------------------------------------
        # PANEL 1: Araştırma Döngüsü Süresi (Gün - Logaritmik)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        days = [
            karsilastirma["arastirma_dongusu_gun"]["1. Human Scientist"],
            karsilastirma["arastirma_dongusu_gun"]["2. Semi-Automated"],
            karsilastirma["arastirma_dongusu_gun"]["3. AI Scientist"],
        ]
        b1 = ax1.bar(modeller, days, color=renkler, width=0.45)
        ax1.set_yscale("log")
        ax1.set_ylabel("Döngü Süresi (Gün - Log Ölçek)", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. Bilimsel Keşif Döngüsü (180 Gün -> 15 Dk | 18,000x)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="y", linestyle=":", alpha=0.3)

        for b, d in zip(b1, days):
            ax1.text(b.get_x() + b.get_width() / 2.0, d * 1.3, f"{d:.2f}g" if d >= 1 else "15 Dk", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 2: Metodolojik Sağlamlık (%)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        soundness = [
            karsilastirma["metodolojik_saglamlik_yuzde"]["1. Human Scientist"],
            karsilastirma["metodolojik_saglamlik_yuzde"]["2. Semi-Automated"],
            karsilastirma["metodolojik_saglamlik_yuzde"]["3. AI Scientist"],
        ]
        b2 = ax2.bar(modeller, soundness, color=renkler, width=0.45)
        ax2.set_ylabel("Metodolojik Sağlamlık (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Deney & Metodoloji Sağlamlığı (%86.5 -> %94.2)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(0, 120)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 2.0, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Otonom Deney Simülasyonu Kayıp Eğrisi
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        epochs = profil_raporu["exp_res"]["epochs"]
        base_loss = profil_raporu["exp_res"]["baseline_loss"]
        prop_loss = profil_raporu["exp_res"]["proposed_loss"]

        ax3.plot(epochs, base_loss, "o--", color="#ef4444", label="Klasik Transformer (Baseline)")
        ax3.plot(epochs, prop_loss, "s-", color="#10b981", linewidth=2.5, label="Önerilen Yöntem (AI Scientist)")
        ax3.set_xlabel("Eğitim Dönemi (Epoch)", fontsize=10, color="#cbd5e1")
        ax3.set_ylabel("Doğrulama Kaybı (Val Loss)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Otonom Deney Yakınsama Grafiği", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.legend(loc="upper right", fontsize=8.5)
        ax3.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 4: NeurIPS / ICLR Hakem Notları (0-10)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        cats = profil_raporu["review_categories"]
        scores = profil_raporu["review_scores"]
        c_colors = ["#10b981", "#38bdf8", "#a855f7", "#10b981"]

        b4 = ax4.bar(cats, scores, color=c_colors, width=0.45)
        ax4.set_ylabel("Hakem Skoru [0 - 10]", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Otonom Hakem Puanları (STRONG ACCEPT)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(0, 12)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha="right")

        for b in b4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 0.2, f"{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9.5)

        # -------------------------------------------------------------
        # PANEL 5: Keşif Başına Ortalama Maliyet ($)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        costs = [
            karsilastirma["maliyet_dolar"]["1. Human Scientist"],
            karsilastirma["maliyet_dolar"]["2. Semi-Automated"],
            karsilastirma["maliyet_dolar"]["3. AI Scientist"],
        ]
        b5 = ax5.bar(modeller, costs, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax5.set_yscale("log")
        ax5.set_ylabel("Maliyet ($ - Log Ölçek)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Araştırma Maliyet Tasarrufu ($50,000 -> $5 | 10,000x)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.grid(axis="y", linestyle=":", alpha=0.3)

        for b, c in zip(b5, costs):
            ax5.text(b.get_x() + b.get_width() / 2.0, c * 1.4, f"${int(c):,}" if c >= 10 else f"${c:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9.5)

        # -------------------------------------------------------------
        # PANEL 6: AI Scientist Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "THE AI SCIENTIST OTONOM ARAŞTIRMA RAPORU\n"
            "====================================================\n"
            "• Mimarî Çerçeve       : The AI Scientist (Sakana AI)\n"
            "• Keşfedilen Hipotez   : Adaptive Sparse Attention (Entropy Gating)\n"
            "• Özgünlük (Novelty)   : %94.0 Doğruluk (arXiv Literatür Taraması)\n"
            "• Deney Başarımı       : Doğruluk: %88.4 -> %96.8 | %58.2 FLOP Tasarruf\n"
            "• Makale Derleme       : NeurIPS Standartlarında Otomatik LaTeX\n"
            "• Otonom Hakemlik      : 9.3/10.0 (STRONG ACCEPT Kararı)\n"
            "• Döngü Hızlanması     : 180 Gün -> 15 Dakika (18,000x Hızlı)\n"
            "• Maliyet Avantajı     : $50,000 -> $5.0 (10,000x Tasarruf)\n"
            "----------------------------------------------------\n"
            "FAZ 15 GÜN 292 AI SCIENTIST TAMAMLANDI!\n"
            "Sırada: Day 293 (Otonom Siber Güvenlik ve Zero-Day Exploit Avcısı)"
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
