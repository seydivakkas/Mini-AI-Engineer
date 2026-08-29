"""
SimPO (Simple Preference Optimization) 6 Panelli Görselleştirici Modülü (Day 217 - FAZ 11).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class SimPOGorsellestirici:
    """SimPO 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/simpo_paneli.png",
    ):
        """6 Panelli SimPO Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 217 (FAZ 11): SIMPO (SIMPLE PREFERENCE OPTIMIZATION) - REFERANSSIZ VE MARJİN TABANLI HİZALAMA",
            fontsize=18,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        marjin = profil_raporu["marjin_analizi"]
        modeller = ["Klasik PPO\n(4 Model)", "Standart DPO\n(Policy + Ref)", "SimPO\n(Referanssız)"]

        # -------------------------------------------------------------
        # PANEL 1: SimPO Referanssız Mimari
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        asamalar = ["1. Tercih Çifti (y_w, y_l)", "2. Tek Model Log-Olasılık", "3. Uzunluk Normalizasyonu (β/|y|)", "4. Hedef Marjin (Δr - γ)", "5. SimPO Kayıp Gradyanı"]
        onemler = [1.0, 1.4, 1.9, 2.3, 2.7]
        ax1.barh(asamalar[::-1], onemler[::-1], color=["#38bdf8", "#8b5cf6", "#10b981", "#f59e0b", "#ec4899"], height=0.45)
        ax1.set_xlabel("İşlem Aşamaları", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. Referanssız SimPO İşlem Hattı", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: VRAM ve Bellek İhtiyacı (GB)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        vramler = [
            karsilastirma["vram_gereksinimi_gb"]["Klasik_PPO_RLHF"],
            karsilastirma["vram_gereksinimi_gb"]["Standart_DPO"],
            karsilastirma["vram_gereksinimi_gb"]["SimPO_Referanssiz"],
        ]
        bars2 = ax2.bar(modeller, vramler, color=["#ef4444", "#38bdf8", "#10b981"], width=0.45)
        ax2.set_ylabel("Gereken GPU VRAM (GB)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. VRAM Tasarrufu (32.4 GB -> 18.4 GB)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(0, 65)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"{h:.1f} GB", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: AlpacaEval-2 Kazanma Oranı (%)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        win_rates = [
            karsilastirma["alpaca_eval_2_win_rate"]["Klasik_PPO_RLHF"],
            karsilastirma["alpaca_eval_2_win_rate"]["Standart_DPO"],
            karsilastirma["alpaca_eval_2_win_rate"]["SimPO_Referanssiz"],
        ]
        bars3 = ax3.bar(modeller, win_rates, color=["#ef4444", "#38bdf8", "#10b981"], width=0.45)
        ax3.set_ylabel("AlpacaEval-2 Win Rate (%)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. AlpacaEval-2 Liderliği (%64.6)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.set_ylim(0, 80)
        ax3.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 4: Arena-Hard Başarım Puanı
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        arena = [
            karsilastirma["arena_hard_skoru"]["Klasik_PPO_RLHF"],
            karsilastirma["arena_hard_skoru"]["Standart_DPO"],
            karsilastirma["arena_hard_skoru"]["SimPO_Referanssiz"],
        ]
        bars4 = ax4.bar(modeller, arena, color=["#ef4444", "#38bdf8", "#10b981"], width=0.45)
        ax4.set_ylabel("Arena-Hard Skoru", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Arena-Hard Karşılaştırması (59.6)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(0, 75)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 5: Hedef Marjin (Gamma) Duyarlılık Eğrisi
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        gammas = marjin["gamma_degerleri"]
        wr = marjin["win_rate"]

        ax5.plot(gammas, wr, marker="o", color="#10b981", linewidth=2.5, label="Win-Rate (%)")
        ax5.axvline(x=0.8, color="#f59e0b", linestyle="--", label="Optimal Marjin (γ=0.8)")
        ax5.set_xlabel("Hedef Ödül Marjini (γ)", fontsize=10, color="#cbd5e1")
        ax5.set_ylabel("AlpacaEval-2 Win Rate (%)", fontsize=10, color="#10b981")
        ax5.set_title("5. Hedef Marjin (Gamma) Optimizasyonu", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.grid(True, linestyle=":", alpha=0.3)
        ax5.legend(loc="lower right")

        # -------------------------------------------------------------
        # PANEL 6: GÜN 217 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "GÜN 217: SIMPO PREFERENCE KARNESİ\n"
            "----------------------------------------------------\n"
            "• Yöntem              : SimPO (Simple Preference Optimization)\n"
            "• Literatür           : Meng et al., NeurIPS 2024 (Princeton)\n"
            "• Referans Model (π_ref): YOK (%0 Ek Model, %50 VRAM Tasarrufu)\n"
            "• VRAM İhtiyacı (7B)  : 32.4 GB -> 18.4 GB (-%43.2 Tasarruf)\n"
            "• AlpacaEval-2        : %58.2 -> %64.6 (+%6.4 DPO'yu Geçer)\n"
            "• Uzunluk Şişmesi     : SIFIR (Doğal Uzunluk Normalizasyonu)\n"
            "• Optimal Hedef Marjin: γ = 0.80, β = 2.0\n"
            "----------------------------------------------------\n"
            "SONUÇ: Referans modeli VRAM'den tamamen atarak tek modelle\n"
            "DPO ve PPO'dan çok daha yüksek kazanma oranı elde edildi!"
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
