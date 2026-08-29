"""
GRPO Teşhis Panosu Görselleştirici Modülü (Day 114).
6 panelli GRPO kayıp eğrileri, akıl yürütme ödül gelişimi, KL cezası, PPO vs GRPO kıyas tablosu ve matematik kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class GRPOGorsellestirici:
    """GRPO analizi için 6 panelli teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        egitim_raporu: Dict[str, List[float]],
        kiyas_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/grpo_reasoning_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "Group Relative Policy Optimization (GRPO - DeepSeek-R1): Critic-Free Akıl Yürütme & Tercih Paneli",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        epoklar = list(range(1, len(egitim_raporu["toplam_kayiplar"]) + 1))

        # -------------------------------------------------------------
        # PANEL 1: GRPO Kayıp Eğrileri
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.plot(epoklar, egitim_raporu["toplam_kayiplar"], color="#4e73df", lw=3.0, marker="o", label="Toplam GRPO Kaybı")
        ax1.plot(epoklar, egitim_raporu["politika_kayiplari"], color="#1cc88a", lw=2.0, linestyle="--", label="Politika Taşıyıcı Kaybı")
        ax1.set_title("1. GRPO Toplam & Politika Kayıp Eğrileri", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epok")
        ax1.set_ylabel("Kayıp (Loss)")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.legend(loc="upper right")

        # -------------------------------------------------------------
        # PANEL 2: Akıl Yürütme Ödül Gelişimi (Mean & Std)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        mean_r = np.array(egitim_raporu["ortalama_oduller"])
        std_r = np.array(egitim_raporu["std_oduller"])
        ax2.plot(epoklar, mean_r, color="#28a745", lw=3.0, marker="^", label="Ortalama Grup Ödülü (Mean r)")
        ax2.fill_between(epoklar, mean_r - std_r, mean_r + std_r, color="#28a745", alpha=0.2, label="Grup İçi Standart Sapma (±σ)")
        ax2.axhline(1.5, color="#e74a3b", linestyle="--", label="Maksimum Olası Ödül (1.5)")
        ax2.set_title("2. Kural Tabanlı Akıl Yürütme Ödül Gelişimi", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epok")
        ax2.set_ylabel("Ödül (Reward)")
        ax2.set_ylim(-0.1, 1.7)
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.legend(loc="lower right")

        # -------------------------------------------------------------
        # PANEL 3: Token Bazlı KL Sapması (Schulman Estimator)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(epoklar, egitim_raporu["kl_kayiplari"], color="#6f42c1", lw=2.5, marker="s", label="KL Sapması D_KL(π_θ || π_ref)")
        ax3.set_title("3. Politika Referans KL Sapması (D_KL)", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Epok")
        ax3.set_ylabel("KL Değeri")
        ax3.grid(True, linestyle="--", alpha=0.7)
        ax3.legend(loc="upper left")

        # -------------------------------------------------------------
        # PANEL 4: Kırpılma Oranı (% Clip Fraction)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.plot(epoklar, egitim_raporu["kirpilma_oranlari"], color="#fd7e14", lw=2.5, marker="d", label="Kırpılma Oranı (%) [|r_t - 1| > 0.2]")
        ax4.axhline(10.0, color="gray", linestyle=":", label="Nominal Eşik (%10)")
        ax4.set_title("4. Taşıyıcı Oran Kırpılma Oranı (%)", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Epok")
        ax4.set_ylabel("Yüzde (%)")
        ax4.set_ylim(-2, 35)
        ax4.grid(True, linestyle="--", alpha=0.7)
        ax4.legend(loc="upper right")

        # -------------------------------------------------------------
        # PANEL 5: PPO vs GRPO Kıyaslama Tablosu
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. PPO vs GRPO Mimari Kıyas Matrisi", fontsize=12, fontweight="bold", pad=10)

        sutunlar = ["Kriter", "PPO (RLHF)", "GRPO (DeepSeek-R1)"]
        veriler = [
            ["Critic Modeli (V)", "Var (70B Param)", "YOK (0 Parametre)"],
            ["Avantaj Tahmini", "GAE-lambda (Gürültülü)", "Grup Z-Score (r-mean)/std"],
            ["Grup Örnekleme", "1-2 Rollout", "G=8, 64, 128 Rollout"],
            ["VRAM Kullanımı", "%100 (4 Model)", "%35 (Policy + Ref)"],
            ["Akıl Yürütme Uyumu", "Düşük / Orta", "Lider (DeepSeek-R1)"],
        ]

        tablo = ax5.table(
            cellText=veriler,
            colLabels=sutunlar,
            loc="center",
            cellLoc="center",
        )
        tablo.auto_set_font_size(False)
        tablo.set_fontsize(8.5)
        tablo.scale(1.15, 1.6)

        for j in range(len(sutunlar)):
            tablo[(0, j)].set_facecolor("#4e73df")
            tablo[(0, j)].set_text_props(color="white", weight="bold")
        for i in range(1, len(veriler) + 1):
            tablo[(i, 2)].set_facecolor("#d4edda")
            tablo[(i, 2)].set_text_props(weight="bold")

        # -------------------------------------------------------------
        # PANEL 6: GRPO Matematik Kartı & R1 Sertifikası
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GRPO Matematik Kartı & DeepSeek-R1 Kararı", fontsize=12, fontweight="bold", pad=10)

        formuller = (
            "[i] Group Relative Policy Optimization (GRPO):\n"
            "--------------------------------------------------\n"
            "1. Grup İçi Göreli Avantaj (Z-Score):\n"
            "   A_i = ( r_i - mean(r) ) / ( std(r) + eps )\n\n"
            "2. Kırpılmış Taşıyıcı Politika Hedefi:\n"
            "   L_clip = min( r_t A_i, clip(r_t, 1-eps, 1+eps) A_i )\n\n"
            "3. Toplam GRPO Hedefi:\n"
            "   J_GRPO = 1/G * sum_i [ L_clip - beta * D_KL(pi || ref) ]\n\n"
            "--------------------------------------------------\n"
            "• Critic Ağı YOK! (Bellek Rollout için kullanılır)\n"
            "• Kural tabanlı doğruluk & <think> format ödülleri\n"
            "• 'Aha Moment' ve Zincirleme Akıl Yürütme Motoru!\n"
            "=================================================="
        )

        ax6.text(
            0.02, 0.5, formuller,
            fontsize=8.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ GRPO Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
