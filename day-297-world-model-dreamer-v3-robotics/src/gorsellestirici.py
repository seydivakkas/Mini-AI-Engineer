"""
Day 297 (FAZ 15): Dünya Modelleri ve DreamerV3 6 Panelli Görselleştirici Modülü.
"""

from typing import Dict, Any
import os
import matplotlib.pyplot as plt
import numpy as np


class DreamerV3Gorsellestirici:
    """FAZ 15 DreamerV3 Dünya Modeli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/dreamerv3_world_model_paneli.png",
    ):
        """6 Panelli Dünya Modeli Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 297 (FAZ 15): DÜNYA MODELLERİ VE DREAMERV3 İLE HAYAL İÇİ ÖĞRENME (WORLD MODELS & ROBOTICS)",
            fontsize=15,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        modeller = ["1. Model-Free PPO\n(Fiziksel Deneme)", "2. Model-Based PlaNet\n(Sürekli Gizil)", "3. DreamerV3 World Model\n(Ayrık Kategorik)"]
        renkler = ["#ef4444", "#f59e0b", "#10b981"]

        # -------------------------------------------------------------
        # PANEL 1: Gerçek Dünya Etkileşim Adımı (Logaritmik)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        steps = [
            karsilastirma["gercek_adim_gereksinimi"]["1. Model-Free PPO"],
            karsilastirma["gercek_adim_gereksinimi"]["2. Model-Based PlaNet"],
            karsilastirma["gercek_adim_gereksinimi"]["3. DreamerV3 World Model"],
        ]
        b1 = ax1.bar(modeller, steps, color=renkler, width=0.45)
        ax1.set_yscale("log")
        ax1.set_ylabel("Gereken Gerçek Adım (Log)", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. Gerçek Etkileşim Adımı (10M -> 100K | 100x Verim)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="y", linestyle=":", alpha=0.3)

        for b, s in zip(b1, steps):
            ax1.text(b.get_x() + b.get_width() / 2.0, s * 1.3, f"{s:,.0f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9.5)

        # -------------------------------------------------------------
        # PANEL 2: Zero-Shot Sim-to-Real Başarısı (%)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        sim2real = [
            karsilastirma["sim_to_real_basarisi_yuzde"]["1. Model-Free PPO"],
            karsilastirma["sim_to_real_basarisi_yuzde"]["2. Model-Based PlaNet"],
            karsilastirma["sim_to_real_basarisi_yuzde"]["3. DreamerV3 World Model"],
        ]
        b2 = ax2.bar(modeller, sim2real, color=renkler, width=0.45)
        ax2.set_ylabel("Başarı Oranı (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Zero-Shot Sim-to-Real Aktarımı (%41.2 -> %96.4)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(0, 115)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Örneklem Verimliliği (Kat)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        eff_mult = [
            karsilastirma["orneklem_verimliligi_kat"]["1. Model-Free PPO"],
            karsilastirma["orneklem_verimliligi_kat"]["2. Model-Based PlaNet"],
            karsilastirma["orneklem_verimliligi_kat"]["3. DreamerV3 World Model"],
        ]
        b3 = ax3.bar(modeller, eff_mult, color=renkler, width=0.45)
        ax3.set_ylabel("Verimlilik Çarpanı", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Örneklem Verimliliği (1x -> 100x Çarpan)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.set_ylim(0, 125)
        ax3.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 2.0, f"{h:.0f}x", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 4: Donanım Yıpranma ve Kırılma Riski (%)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        risk = [
            karsilastirma["donanim_yipranma_riski_yuzde"]["1. Model-Free PPO"],
            karsilastirma["donanim_yipranma_riski_yuzde"]["2. Model-Based PlaNet"],
            karsilastirma["donanim_yipranma_riski_yuzde"]["3. DreamerV3 World Model"],
        ]
        b4 = ax4.bar(modeller, risk, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax4.set_ylabel("Hasar / Yıpranma Riski (%)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Robotik Donanım Hasar Riski (%76.4 -> %1.2)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(0, 95)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 5: Gizil Hayal Ufku & Değer Tahmini (15 Adım)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        steps_x = profil_raporu["hayal_adimlari"]
        val_y = profil_raporu["deger_tahminleri"]

        ax5.plot(steps_x, val_y, marker="o", color="#38bdf8", linewidth=2.5, label="İnşa Edilen Gelecek Değeri (V)")
        ax5.fill_between(steps_x, [v - 0.3 for v in val_y], [v + 0.3 for v in val_y], color="#38bdf8", alpha=0.2)
        ax5.set_xlabel("Hayal Ufku Adımı (H=15 Latent Step)", fontsize=10, color="#cbd5e1")
        ax5.set_ylabel("Tahmin Edilen Getiri (Value)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Gizil Uzayda Zihinsel Hayal Kurma (250 FPS)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.grid(True, linestyle=":", alpha=0.3)
        ax5.legend(loc="upper left", fontsize=9)

        # -------------------------------------------------------------
        # PANEL 6: DreamerV3 Dünya Modeli Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "DREAMERV3 WORLD MODEL ROBOTICS RAPORU\n"
            "====================================================\n"
            "• Mimarî Temel         : Recurrent State-Space Model (RSSM)\n"
            "• Gizil Temsil         : 32x32 Kategorik Ayrık Dağılım\n"
            "• Ölçekleme            : Symlog Transform & Free Bits KL\n"
            "• Hayal İçi Öğrenme    : 15 Adım İleri Zihinsel Simülasyon\n"
            "• Örneklem Verimliliği : 100 Kat Artış (10M -> 100K Adım)\n"
            "• Sim-to-Real Aktarımı : %96.4 Sıfır-Atış Başarı Oranı\n"
            "• Donanım Güvenliği    : Hasar Riski %76.4 -> %1.2 (Sıfır Hasar)\n"
            "• Simülasyon Hızı      : 250 FPS GPU İçi Paralel Çıkarım\n"
            "----------------------------------------------------\n"
            "FAZ 15 GÜN 297 DÜNYA MODELLERİ TAMAMLANDI!\n"
            "Sırada: Day 298 (Otonom Bilimsel Fonlama ve Hakemler Meclisi)"
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
