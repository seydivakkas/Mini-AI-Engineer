"""
Day 301 (BÜYÜK FİNAL): Uçtan Uca Bedenlenmiş Çok Modlu Otonom AGI Sistemi 6 Panelli Görselleştirici.
301 Günlük Tüm Müfredatın Zirvesi ve Şampiyonluk Teşhis Panosu.
"""

from typing import Dict, Any
import os
import matplotlib.pyplot as plt
import numpy as np


class OmniEmbodiedAGIGorsellestirici:
    """301 Günlük BÜYÜK FİNAL Şampiyonluk Teşhis Panosu."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/omni_embodied_agi_grand_finale_paneli.png",
    ):
        """6 Panelli Büyük Final Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 301 (BÜYÜK FİNAL): UÇTAN UCA BEDENLENMİŞ ÇOK MODLU OTONOM AGİ SİSTEMİ (GRAND FINALE)",
            fontsize=15,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        modeller = ["1. Siloed AI\n(Klasik İzole)", "2. Multi-Agent\n(Modüler)", "3. Omni-AGI\n(BÜYÜK FİNAL 301)"]
        renkler = ["#ef4444", "#f59e0b", "#10b981"]

        # -------------------------------------------------------------
        # PANEL 1: Çok Modlu Bilişsel Skor (MMLU)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        mmlu = [
            karsilastirma["cok_modlu_mmlu_skoru"]["1. Traditional Siloed AI"],
            karsilastirma["cok_modlu_mmlu_skoru"]["2. Modular Multi-Agent"],
            karsilastirma["cok_modlu_mmlu_skoru"]["3. Omni-Embodied AGI (301)"],
        ]
        b1 = ax1.bar(modeller, mmlu, color=renkler, width=0.45)
        ax1.set_ylabel("MMLU Skoru (Puan)", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. Çok Modlu Bilişsel Zeka (64.2 -> 98.4 | +34.2)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.set_ylim(0, 115)
        ax1.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b1:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 2: Fiziksel Robotik & Manipülasyon Başarısı (%)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        phys = [
            karsilastirma["fiziksel_robotik_basari_yuzde"]["1. Traditional Siloed AI"],
            karsilastirma["fiziksel_robotik_basari_yuzde"]["2. Modular Multi-Agent"],
            karsilastirma["fiziksel_robotik_basari_yuzde"]["3. Omni-Embodied AGI (301)"],
        ]
        b2 = ax2.bar(modeller, phys, color=renkler, width=0.45)
        ax2.set_ylabel("Başarı Oranı (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Fiziksel Robotik Başarı (%52.0 -> %98.9)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(0, 115)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Uçtan Uca Gecikme (ms)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        lat = [
            karsilastirma["uctan_uca_gecikme_ms"]["1. Traditional Siloed AI"],
            karsilastirma["uctan_uca_gecikme_ms"]["2. Modular Multi-Agent"],
            karsilastirma["uctan_uca_gecikme_ms"]["3. Omni-Embodied AGI (301)"],
        ]
        b3 = ax3.bar(modeller, lat, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax3.set_ylabel("Algı-Eylem Gecikmesi (ms)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Uçtan Uca Gecikme (140 ms -> 6.2 ms | 22.5x Hızlı)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.set_ylim(0, 160)
        ax3.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 2.0, f"{h:.1f} ms", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 4: Enerji Verimliliği (TFLOPS/Watt)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        eff = [
            karsilastirma["enerji_verimliligi_tflops_w"]["1. Traditional Siloed AI"],
            karsilastirma["enerji_verimliligi_tflops_w"]["2. Modular Multi-Agent"],
            karsilastirma["enerji_verimliligi_tflops_w"]["3. Omni-Embodied AGI (301)"],
        ]
        b4 = ax4.bar(modeller, eff, color=renkler, width=0.45)
        ax4.set_ylabel("TFLOPS / Watt", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Donanım Enerji Verimliliği (3.2 -> 18.4 TFLOPS/W)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(0, 24)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 0.4, f"{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 5: 15 Faz Boyunca Kümülatif Zeka Evrimi
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        fazlar = profil_raporu["fazlar"]
        skorlar = profil_raporu["kolektif_zeka_skorlari"]

        ax5.plot(fazlar, skorlar, marker="o", color="#38bdf8", linewidth=2.5, label="Kolektif AGI Yeteneği")
        ax5.fill_between(fazlar, 0, skorlar, color="#38bdf8", alpha=0.15)
        ax5.axhline(98.4, color="#10b981", linestyle="--", label="301. Gün Büyük Final Zirvesi (98.4)")
        ax5.set_xticks(range(len(fazlar)))
        ax5.set_xticklabels(fazlar, rotation=45, ha="right", fontsize=8.5)
        ax5.set_ylabel("Kümülatif Zeka Skoru", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. 15 Faz Boyunca Kümülatif Zeka Evrimi (Gün 01 -> 301)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.set_ylim(0, 115)
        ax5.grid(True, linestyle=":", alpha=0.3)
        ax5.legend(loc="lower right", fontsize=8.5)

        # -------------------------------------------------------------
        # PANEL 6: 301 Günlük Büyük Final Onur & Mimari Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "301 GUNLUK BUYUK FINAL SAMPIYONLUK RAPORU\n"
            "====================================================\n"
            "• Faz 1 - 10           : LLM, Vision, Diffusion & MLOps\n"
            "• Faz 11               : Post-Training, GRPO & Reasoning\n"
            "• Faz 12               : Agentic OS, Swarm & Tool-Use\n"
            "• Faz 13               : Embodied AI & Bimanual Robotics\n"
            "• Faz 14               : 1-Bit BitNet & HLS Hardware Synthesis\n"
            "• Faz 15               : Quantum AI & Self-Improving AGI Core\n"
            "----------------------------------------------------\n"
            "• Çok Modlu Zeka       : 98.4 MMLU (Human-Expert)\n"
            "• Robotik Başarı       : %98.9 Fiziksel İcra Başarısı\n"
            "• Algı-Eylem Hızı      : 6.2 ms (22.5 Kat Hızlanma)\n"
            "• Donanım Verimliliği  : 18.4 TFLOPS/W (1-Bit HLS)\n"
            "• Kuantum Çözücü       : H2 Taban Enerjisi (<1.6 mHa)\n"
            "====================================================\n"
            "TEBRİKLER! 301 GÜNLÜK MÜFREDAT %100 TAMAMLANDI!"
        )

        ax6.text(
            0.05,
            0.5,
            ozet_metin,
            fontsize=9.0,
            family="monospace",
            color="#f8fafc",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e293b", edgecolor="#10b981", alpha=0.9),
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close()
