"""
GRPO Matematiksel Akıl Yürütme 6 Panelli Görselleştirici Modülü (Day 202 - FAZ 11).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class GRPOGorsellestirici:
    """GRPO 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/grpo_math_reasoning_paneli.png",
    ):
        """6 Panelli GRPO Matematiksel Akıl Yürütme Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 202 (FAZ 11): GRPO (GROUP RELATIVE POLICY OPTIMIZATION) İLE MATEMATİKSEL AKIL YÜRÜTME",
            fontsize=18,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        adimlar = profil_raporu["adimlar"]
        dogruluklar = profil_raporu["dogruluk_oranlari"]
        uzunluklar = profil_raporu["dusunce_uzunluklari"]

        # -------------------------------------------------------------
        # PANEL 1: GRPO Mimarisi ve Akış Boru Hattı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        bloklar = ["1. Matematik Sorusu", "2. Grup Örneklemesi (G=4)", "3. Kural Tabanlı Doğrulayıcı", "4. Grup İçi Bağıl Avantaj", "5. Critic'siz Politika Güncellemesi"]
        onem = [1.0, 1.4, 1.8, 2.2, 2.0]
        ax1.barh(bloklar[::-1], onem[::-1], color=["#38bdf8", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"], height=0.45)
        ax1.set_xlabel("Akış Hiyerarşisi", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. DeepSeek-R1 GRPO Mimari Akışı", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: Eğitim Sürecinde Matematik Doğruluk Artışı (%)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(adimlar, dogruluklar, marker="o", color="#10b981", lw=2.2, label="Doğruluk Oranı (%)")
        ax2.set_xlabel("GRPO Eğitim Adımı", fontsize=10, color="#cbd5e1")
        ax2.set_ylabel("Matematik Doğruluk Skoru (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title(f"2. Doğruluk Kazanımı (%30 -> %{profil_raporu['son_dogruluk']:.1f})", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(20, 105)
        ax2.legend(loc="lower right", fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 3: Standart PPO vs GRPO Bellek & Hız Kıyası
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        metotlar = ["Standart PPO\n(Actor + 70B Critic)", "DeepSeek GRPO\n(Sıfır Critic Modeli)"]
        vram_kullanim = [100.0, 50.0]  # Göreli VRAM
        bars3 = ax3.bar(metotlar, vram_kullanim, color=["#ef4444", "#10b981"], width=0.45)
        ax3.set_ylabel("Gereken GPU VRAM Oranı (%)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. PPO vs GRPO Bellek Tasarrufu (%50 VRAM)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.grid(axis="y", linestyle=":", alpha=0.4)

        for b in bars3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.0f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 4: Düşünce Zinciri (CoT) Token Uzunluğu Evrimi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        bars4 = ax4.bar(adimlar, uzunluklar, color="#8b5cf6", width=0.55)
        ax4.set_xlabel("Eğitim Adımı", fontsize=10, color="#cbd5e1")
        ax4.set_ylabel("Ortalama <think> Token Sayısı", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Düşünce Uzunluğu Artışı (Aha-Moment)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 5: Grup İçi Ödül ve Bağıl Avantaj Ayrışması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        adaylar = ["Aday 1\n(Doğru+Format)", "Aday 2\n(Yanlış+Format)", "Aday 3\n(Yanlış)", "Aday 4\n(Yanlış)"]
        avantaj_degerleri = [1.50, -0.25, -0.62, -0.62]
        renkler = ["#10b981", "#f59e0b", "#ef4444", "#ef4444"]
        bars5 = ax5.bar(adaylar, avantaj_degerleri, color=renkler, width=0.45)
        ax5.axhline(0.0, color="#ffffff", linestyle="-", lw=1.0)
        ax5.set_ylabel("Grup İçi Bağıl Avantaj (A_i)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Örnek G=4 Grup İçi Avantaj Standardizasyonu", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.grid(axis="y", linestyle=":", alpha=0.4)

        for b in bars5:
            h = b.get_height()
            va = "bottom" if h >= 0 else "top"
            ax5.text(b.get_x() + b.get_width() / 2.0, h + (0.05 if h >= 0 else -0.15), f"{h:+.2f}", ha="center", va=va, color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 6: GÜN 202 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "GÜN 202: GRPO MATEMATİKSEL AKIL YÜRÜTME KARNESİ\n"
            "----------------------------------------------------\n"
            "• Algoritma           : Group Relative Policy Optimization (GRPO)\n"
            "• Öncü Referans       : DeepSeekMath & DeepSeek-R1\n"
            "• Critic Ağı İhtiyacı : %0 (Sıfır Değer Modeli / Zero Critic)\n"
            "• Bellek Tasarrufu    : %50 VRAM Tasarrufu (PPO'ya Kıyasla)\n"
            "• Eğitim Hızlanması   : 2.1x Daha Yüksek Throughput\n"
            "• Ödül Mekanizması    : Kural Tabanlı (Biçim + Kesin Sayısal Eşleşme)\n"
            f"• Son Doğruluk Seviyesi: %{profil_raporu['son_dogruluk']:.1f}\n"
            "----------------------------------------------------\n"
            "SONUÇ: Ayrı bir Critic modeli eğitmeden doğrudan grup içi\n"
            "ödül karşılaştırmasıyla LLM akıl yürütmesi güçlendirildi!"
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
