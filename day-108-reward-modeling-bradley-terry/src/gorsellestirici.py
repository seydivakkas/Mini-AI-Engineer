"""
Bradley-Terry Ödül Modeli Teşhis Panosu Görselleştirici Modülü (Day 108).
6 panelli kayıp eğrisi, ikili tercih doğruluğu, marjin ayrışması ve sigmoid olasılık panosu.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class OdulGorsellestirici:
    """Bradley-Terry Reward Model analizi için 6 panelli teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        egitim_raporu: Dict[str, List[float]],
        hacking_raporu: Dict[str, float],
        kayit_yolu: str = "ciktilar/bradley_terry_reward_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "Bradley-Terry Tercih Modeli, Skaler Ödül Fonksiyonu & Reward Hacking Teşhis Paneli",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        epoklar = list(range(1, len(egitim_raporu["kayiplar"]) + 1))

        # -------------------------------------------------------------
        # PANEL 1: Bradley-Terry Kayıp Eğrisi
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.plot(epoklar, egitim_raporu["kayiplar"], color="#e74a3b", lw=2.5, marker="o", label="Bradley-Terry Loss")
        ax1.set_title("1. Bradley-Terry Kayıp Eğrisi (-log(sigmoid(r_w - r_l)))", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Eğitim Epoku")
        ax1.set_ylabel("Kayıp (Loss)")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.legend(loc="upper right")

        # -------------------------------------------------------------
        # PANEL 2: İkili Sıralama Doğruluğu (%)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(epoklar, egitim_raporu["dogruluklar"], color="#1cc88a", lw=3.0, marker="s", label="Tercih Doğruluğu (r_w > r_l)")
        ax2.axhline(50.0, color="gray", linestyle=":", label="Rastgele Tahmin (%50)")
        ax2.set_title("2. Çiftli Sıralama Doğruluğu (% Accuracy)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Eğitim Epoku")
        ax2.set_ylabel("Doğruluk (%)")
        ax2.set_ylim(40, 105)
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.legend(loc="lower right")

        # -------------------------------------------------------------
        # PANEL 3: Ödül Marjin Ayrışması (r_w vs r_l)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(epoklar, egitim_raporu["r_w_ort"], color="#4e73df", lw=2.5, marker="^", label="Ortalama r_chosen (r_w)")
        ax3.plot(epoklar, egitim_raporu["r_l_ort"], color="#e74a3b", lw=2.5, marker="v", label="Ortalama r_rejected (r_l)")
        ax3.fill_between(epoklar, egitim_raporu["r_l_ort"], egitim_raporu["r_w_ort"], color="#4e73df", alpha=0.15, label="Ödül Ayrışma Marjini")
        ax3.set_title("3. Ödül Puanlarının Epoklar Boyunca Ayrışması", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Eğitim Epoku")
        ax3.set_ylabel("Skaler Ödül Puanı")
        ax3.grid(True, linestyle="--", alpha=0.7)
        ax3.legend(loc="center left")

        # -------------------------------------------------------------
        # PANEL 4: Bradley-Terry Sigmoid Olasılık Fonksiyonu
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        delta_r = np.linspace(-5, 5, 200)
        prob = 1.0 / (1.0 + np.exp(-delta_r))
        ax4.plot(delta_r, prob, color="#6f42c1", lw=3.0, label=r"$P(y_w \succ y_l) = \sigma(\Delta r)$")
        ax4.axvline(0.0, color="gray", linestyle="--")
        ax4.axhline(0.5, color="gray", linestyle=":")
        son_marjin = egitim_raporu["marjinler"][-1]
        son_prob = 1.0 / (1.0 + np.exp(-son_marjin))
        ax4.scatter([son_marjin], [son_prob], color="red", s=100, zorder=5, label=rf"Mevcut Model: $\Delta r$={son_marjin:.2f} (P={son_prob*100:.1f}%)")

        ax4.set_title("4. Bradley-Terry Olasılık ve Karar Eşiği", fontsize=12, fontweight="bold")
        ax4.set_xlabel(r"Ödül Farkı $\Delta r = r_w - r_l$")
        ax4.set_ylabel("Kazanma Olasılığı")
        ax4.grid(True, linestyle="--", alpha=0.7)
        ax4.legend(loc="upper left")

        # -------------------------------------------------------------
        # PANEL 5: Bradley-Terry ve Skaler Başlık Formül Kartı
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Bradley-Terry Ödül Matematik Kartı", fontsize=12, fontweight="bold", pad=10)

        formuller = (
            "[i] Bradley-Terry Tercih Modellemesi:\n"
            "--------------------------------------------------\n"
            "1. İkili Tercih Olasılığı:\n"
            "   P(y_w > y_l | x) = sigma(r_psi(x, y_w) - r_psi(x, y_l))\n\n"
            "2. Marjin Destekli Negatif Log-Likelihood Kaybı:\n"
            "   Loss = -E[ log sigma(r_w - r_l - margin) ] + lambda * ||r||^2\n\n"
            "3. Skaler Ödül Başlığı (Scalar Score Head):\n"
            "   h_last = Transformer(x, y)[:, -1, :]\n"
            "   r(x, y) = W_score^T * h_last (1D Skaler Puan)"
        )

        ax5.text(
            0.05, 0.5, formuller,
            fontsize=8.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: Stajyer Notu & Reward Model Karar Sertifikası
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. Stajyer Notu & Reward Model Karar Sertifikası", fontsize=12, fontweight="bold", pad=10)

        sertifika = (
            "====================================================\n"
            "         REWARD MODELING DECISION CERTIFICATE       \n"
            "====================================================\n"
            "• Soru: Neden direkt 1-10 puan vermek yerine çiftli? \n"
            "• Cevap: İnsanlar ve LLM'ler mutlak puanlamada tutar-\n"
            "         sızdır, ancak A > B kıyaslamasında %95+    \n"
            "         tutarlıdır (Bradley-Terry gücü)!\n"
            f"• Reward Hacking Ayrışma Güveni: {hacking_raporu['ayrisma_guvenilirligi']:.2f} Puan\n"
            "----------------------------------------------------\n"
            "[ONAYLANDI] PPO / RLHF / DPO Ön Koşul Standardı!\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, sertifika,
            fontsize=8.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=2.0),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Bradley-Terry Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
