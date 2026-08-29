"""
DPO Tercih Hizalaması Teşhis Panosu Görselleştirici Modülü (Day 110).
6 panelli DPO kayıp eğrisi, örtük ödül ayrışması, doğruluk, DPO vs PPO kıyası ve matematik kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class DPOGorsellestirici:
    """DPO analizi için 6 panelli teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        egitim_raporu: Dict[str, List[float]],
        kiyasi_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/dpo_alignment_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "Direct Preference Optimization (DPO): Reward Modelsiz Kapalı Form Tercih Hizalama Paneli",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        epoklar = list(range(1, len(egitim_raporu["kayiplar"]) + 1))

        # -------------------------------------------------------------
        # PANEL 1: DPO Kayıp Eğrisi
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.plot(epoklar, egitim_raporu["kayiplar"], color="#4e73df", lw=2.5, marker="o", label="DPO Kaybı")
        ax1.set_title("1. DPO Log-Oranı Kayıp Eğrisi (Epoklar Boyunca)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epok")
        ax1.set_ylabel("L_DPO Kayıp Değeri")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.legend(loc="upper right")

        # -------------------------------------------------------------
        # PANEL 2: Çiftli Tercih Doğruluğu (% Accuracy)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(epoklar, egitim_raporu["dogruluklar"], color="#1cc88a", lw=3.0, marker="s", label="Doğruluk (%)")
        ax2.axhline(50.0, color="gray", linestyle=":", label="Rastgele Tahmin (%50)")
        ax2.axhline(95.0, color="#e74a3b", linestyle="--", label="Hedef Eşik (%95)")
        ax2.set_title("2. Çiftli Sıralama Doğruluğu (Pairwise Preference Accuracy)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epok")
        ax2.set_ylabel("Doğruluk (%)")
        ax2.set_ylim(40, 105)
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.legend(loc="lower right")

        # -------------------------------------------------------------
        # PANEL 3: Örtük Ödüllerin Ayrışması (r_w vs r_l & Delta r)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(epoklar, egitim_raporu["r_w_ort"], color="#28a745", lw=2.5, marker="^", label=r"$\hat{r}_w$ (Chosen Örtük Ödül)")
        ax3.plot(epoklar, egitim_raporu["r_l_ort"], color="#dc3545", lw=2.5, marker="v", label=r"$\hat{r}_l$ (Rejected Örtük Ödül)")
        ax3.plot(epoklar, egitim_raporu["marjinler"], color="#6f42c1", lw=2.0, linestyle="--", label=r"$\Delta \hat{r} = \hat{r}_w - \hat{r}_l$")
        ax3.set_title("3. Örtük Ödüllerin ve Marjinin Ayrışması", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Epok")
        ax3.set_ylabel("Örtük Ödül (Implicit Reward)")
        ax3.grid(True, linestyle="--", alpha=0.7)
        ax3.legend(loc="center left")

        # -------------------------------------------------------------
        # PANEL 4: DPO vs PPO Bellek ve Karmaşıklık Kıyası
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        kategoriler = ["PPO (RLHF)", "DPO (Direct)"]
        modeller = [kiyasi_raporu["ppo_model_sayisi"], kiyasi_raporu["dpo_model_sayisi"]]
        renkler = ["#e74a3b", "#36b9cc"]
        barlar = ax4.bar(kategoriler, modeller, color=renkler, width=0.55, edgecolor="black")

        for bar, val in zip(barlar, modeller):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val} Model\n({100 if val==4 else 50}% VRAM)",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax4.set_title("4. DPO vs PPO Model Sayısı ve VRAM Tasarrufu (%50 Kazanç)", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Eşzamanlı Bellekteki Model Sayısı")
        ax4.set_ylim(0, 5)
        ax4.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 5: DPO Matematik Kartı
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. DPO Kapalı Form Matematik Kartı", fontsize=12, fontweight="bold", pad=10)

        formuller = (
            "[i] DPO Kapalı Form Tercih Hizalaması:\n"
            "--------------------------------------------------\n"
            "1. Bradley-Terry Ödül Formülasyonu:\n"
            "   r(x,y) = beta * log( pi_theta(y|x) / pi_ref(y|x) )\n\n"
            "2. DPO Kayıp Fonksiyonu (L_DPO):\n"
            "   L_DPO = -E [ log sigma( beta * log(pi/ref_w) - beta * log(pi/ref_l) ) ]\n\n"
            "3. Gradyan Dinamiği:\n"
            "   grad = -beta * sigma(r_l - r_w) * [ grad log pi(y_w) - grad log pi(y_l) ]\n\n"
            "-> Critic ve Reward Model tamamen ortadan kalkar!"
        )

        ax5.text(
            0.05, 0.5, formuller,
            fontsize=8.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: Stajyer Notu & DPO Karar Sertifikası
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. Stajyer Notu & DPO Karar Sertifikası", fontsize=12, fontweight="bold", pad=10)

        sertifika = (
            "====================================================\n"
            "            DPO ALIGNMENT CERTIFICATE               \n"
            "====================================================\n"
            "• Soru: Neden DPO günümüzde PPO'nun yerini aldı?    \n"
            "• Cevap: DPO, takviyeli öğrenmeyi (RL) doğrudan     \n"
            "         Supervised Fine-Tuning basitliğine indirger.\n"
            "         Reward Hacking ve PPO kararsızlığı biter!  \n"
            "----------------------------------------------------\n"
            "[ONAYLANDI] Llama 3, Mistral & Zephyr Tercih Modeli!\n"
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
        print(f"  ✓ DPO Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
