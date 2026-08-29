"""
PPO Actor-Critic ve RLHF Hizalama Teşhis Panosu Görselleştirici Modülü (Day 109).
6 panelli ödül artışı, KL sapması, kırpma oranı, kayıplar ve mimari teşhis panosu.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class PPOGorsellestirici:
    """PPO Actor-Critic analizi için 6 panelli teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        egitim_raporu: Dict[str, List[float]],
        kayit_yolu: str = "ciktilar/ppo_actor_critic_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "PPO ile LLM Hizalama: 4-Modelli Actor-Critic, GAE ve KL Divergence Penalty Analiz Paneli",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        adimlar = list(range(1, len(egitim_raporu["oduller"]) + 1))

        # -------------------------------------------------------------
        # PANEL 1: Ortalama Ödül Gelişimi
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.plot(adimlar, egitim_raporu["oduller"], color="#1cc88a", lw=3.0, marker="o", label="Ortalama Model Ödülü")
        ax1.set_title("1. PPO Optimizasyonu ile Model Ödül Artışı", fontsize=12, fontweight="bold")
        ax1.set_xlabel("PPO Eğitim Adımı")
        ax1.set_ylabel("Skaler Ödül Puanı")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.legend(loc="lower right")

        # -------------------------------------------------------------
        # PANEL 2: Token Bazlı KL Divergence Sapması
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(adimlar, egitim_raporu["kl_sapmalari"], color="#e74a3b", lw=2.5, marker="s", label="KL Divergence (D_KL)")
        ax2.axhline(0.2, color="orange", linestyle="--", label="Güvenli KL Sınırı")
        ax2.set_title("2. Referans Modelden Sapma (KL Penalty Kontrolü)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("PPO Eğitim Adımı")
        ax2.set_ylabel("KL Sapması")
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.legend(loc="upper left")

        # -------------------------------------------------------------
        # PANEL 3: PPO Kırpma Oranı (% Clip Fraction)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(adimlar, egitim_raporu["kirpma_oranlari"], color="#4e73df", lw=2.5, marker="^", label="Kırpılan Token Oranı (%)")
        ax3.fill_between(adimlar, 0, egitim_raporu["kirpma_oranlari"], color="#4e73df", alpha=0.15)
        ax3.set_title("3. PPO Politika Kırpma Oranı (% Clip Fraction, eps=0.2)", fontsize=12, fontweight="bold")
        ax3.set_xlabel("PPO Eğitim Adımı")
        ax3.set_ylabel("Kırpılma Oranı (%)")
        ax3.grid(True, linestyle="--", alpha=0.7)
        ax3.legend(loc="upper right")

        # -------------------------------------------------------------
        # PANEL 4: Politika ve Değer Ağı Kayıp Eğrileri
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.plot(adimlar, egitim_raporu["politika_kayiplari"], color="#6f42c1", lw=2.0, label="Politika Kaybı (Actor)")
        ax4.plot(adimlar, egitim_raporu["deger_kayiplari"], color="#f6c23e", lw=2.0, label="Değer Kaybı (Critic MSE)")
        ax4.set_title("4. Actor ve Critic Kayıp Eğrileri", fontsize=12, fontweight="bold")
        ax4.set_xlabel("PPO Eğitim Adımı")
        ax4.set_ylabel("Kayıp Değeri")
        ax4.grid(True, linestyle="--", alpha=0.7)
        ax4.legend(loc="upper right")

        # -------------------------------------------------------------
        # PANEL 5: 4-Modelli PPO Matematik Kartı
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. PPO & GAE Matematiksel Formül Kartı", fontsize=12, fontweight="bold", pad=10)

        formuller = (
            "[i] PPO 4-Modelli RLHF Mimarisi:\n"
            "--------------------------------------------------\n"
            "1. KL Cezalı Birleşik Ödül:\n"
            "   R_t = -beta * (log pi_theta - log pi_ref) + delta_tT * R_RM\n\n"
            "2. Generalized Advantage Estimation (GAE-lambda):\n"
            "   delta_t = R_t + gamma * V(s_t+1) - V(s_t)\n"
            "   A_t = delta_t + gamma * lambda * A_t+1\n\n"
            "3. PPO Kırpılmış Politika Kaybı:\n"
            "   L_CLIP = -min( r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t )"
        )

        ax5.text(
            0.05, 0.5, formuller,
            fontsize=8.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: Stajyer Notu & PPO Karar Sertifikası
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. Stajyer Notu & PPO Karar Sertifikası", fontsize=12, fontweight="bold", pad=10)

        sertifika = (
            "====================================================\n"
            "         PPO RLHF ALIGNMENT CERTIFICATE             \n"
            "====================================================\n"
            "• Soru: Neden 4 ayrı model aynı anda GPU'dadır?    \n"
            "• Cevap: Aktör metin üretir, Eleştirmen değer bi-  \n"
            "         çer, Ödül Modeli insan beğenisi verir,     \n"
            "         Ref Model ise aktörün sapıtmasını önler!  \n"
            "----------------------------------------------------\n"
            "[ONAYLANDI] ChatGPT & InstructGPT Hizalama Zirvesi!\n"
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
        print(f"  ✓ PPO Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
