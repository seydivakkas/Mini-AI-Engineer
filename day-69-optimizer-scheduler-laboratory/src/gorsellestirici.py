"""
Optimizer ve Scheduler Laboratuvari Gorsellestiricisi
=====================================================
AdamW, Lion ve Scheduler deneylerinin egitim kaybi, dogrulama basarimi,
LR cizelgesi, gradyan normu ve SWOT matrisini 6 panelli yuksek cozunurluklu endustriyel tabloda birlestirir.
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class OptimizerLaboratuvarGorsellestirici:
    """
    Day 69 6-Panelli Optimizasyon ve Zamanlama Teşhis Panosu üreticisi.
    """

    @staticmethod
    def panoyu_ciz_ve_kaydet(
        laboratuvar_sonuclari: Dict[str, Any],
        cikti_yolu: str = "ciktilar/optimizer_karsilastirma_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(21, 13))
        fig.suptitle(
            "Day 69: AdamW vs Lion Optimizer, CosineAnnealing & Linear Warmup Laboratuvar Paneli",
            fontsize=17,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        d1 = laboratuvar_sonuclari["deney_1"]
        d2 = laboratuvar_sonuclari["deney_2"]
        d3 = laboratuvar_sonuclari["deney_3"]
        toplam_epoch = laboratuvar_sonuclari["toplam_epoch"]
        epochs = list(range(1, toplam_epoch + 1))

        # -------------------------------------------------------------
        # 1. Panel: Yönetici & Optimizasyon Özet Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        ozet_metin = (
            "       OPTIMIZER & SCHEDULER LABORATUVARI OZETI\n"
            "═══════════════════════════════════════════════════════\n"
            f" 1. {d1['deney_adi']:<22} : Loss: {d1['son_train_loss']:<7.4f} | Acc: %{d1['son_val_accuracy']:.1f} | Opt Mem: {d1['tahmini_opt_bellek_kb']:.0f} KB\n"
            f" 2. {d2['deney_adi']:<22} : Loss: {d2['son_train_loss']:<7.4f} | Acc: %{d2['son_val_accuracy']:.1f} | Opt Mem: {d2['tahmini_opt_bellek_kb']:.0f} KB\n"
            f" 3. {d3['deney_adi']:<22} : Loss: {d3['son_train_loss']:<7.4f} | Acc: %{d3['son_val_accuracy']:.1f} | Opt Mem: {d3['tahmini_opt_bellek_kb']:.0f} KB\n"
            "───────────────────────────────────────────────────────\n"
            " * Lion Bellek Tasarrufu : %50 DAHA AZ GPU RAM (v_t yok)\n"
            " * Warmup + Cosine       : Aşırı gradyan şokunu engelledi\n"
            " * Decoupled Decay (WD)  : Bias/Norm hariç tutuldu (Best Practice)\n"
            "═══════════════════════════════════════════════════════\n"
            " * Google Brain Lion (AutoML Evolved Optimizer): AKTIF"
        )
        ax1.text(
            0.5, 0.5, ozet_metin,
            transform=ax1.transAxes,
            fontsize=10.0,
            family="monospace",
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=1.2", facecolor="#e8f8f5", edgecolor="#1abc9c", linewidth=2.0)
        )
        ax1.set_title("1. Optimizasyon Laboratuvar Ozeti", fontweight="bold", color="#16a085")

        # -------------------------------------------------------------
        # 2. Panel: Eğitim Kaybı (Train Loss) Yakınsama Eğrileri
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(epochs, d1["gecmis"]["train_loss"], label=f"{d1['deney_adi']}", color="#e74c3c", linestyle="--", linewidth=2.0)
        ax2.plot(epochs, d2["gecmis"]["train_loss"], label=f"{d2['deney_adi']}", color="#3498db", linewidth=2.5)
        ax2.plot(epochs, d3["gecmis"]["train_loss"], label=f"{d3['deney_adi']}", color="#2ecc71", linewidth=2.5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Egitim Kaybi (Cross-Entropy)")
        ax2.set_title("2. Egitim Kaybi Yakinsama Dinamigi", fontweight="bold", color="#2980b9")
        ax2.legend(loc="upper right")

        # -------------------------------------------------------------
        # 3. Panel: Doğrulama Başarımı (Validation Accuracy %)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(epochs, d1["gecmis"]["val_accuracy"], label=f"{d1['deney_adi']}", color="#e74c3c", marker="x", linestyle="--", linewidth=1.8)
        ax3.plot(epochs, d2["gecmis"]["val_accuracy"], label=f"{d2['deney_adi']}", color="#3498db", marker="o", linewidth=2.2)
        ax3.plot(epochs, d3["gecmis"]["val_accuracy"], label=f"{d3['deney_adi']}", color="#2ecc71", marker="s", linewidth=2.2)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Dogruluk / Accuracy (%)")
        ax3.set_title("3. Dogrulama Basarimi Karsilastirmasi", fontweight="bold", color="#8e44ad")
        ax3.legend(loc="lower right")

        # -------------------------------------------------------------
        # 4. Panel: Öğrenme Oranı (LR Schedule Dynamics)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.plot(epochs, d1["gecmis"]["lr"], label="AdamW (StepLR)", color="#e74c3c", linestyle="--", linewidth=2.0)
        ax4.plot(epochs, d2["gecmis"]["lr"], label="AdamW (WarmupCosine)", color="#3498db", linewidth=2.5)
        ax4.plot(epochs, [x * 10 for x in d3["gecmis"]["lr"]], label="Lion (WarmupCosine x10 olcek)", color="#2ecc71", linestyle=":", linewidth=2.5)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Ogrenme Orani (LR)")
        ax4.set_title("4. Linear Warmup + Cosine LR Profili", fontweight="bold", color="#d35400")
        ax4.legend(loc="upper right")

        # -------------------------------------------------------------
        # 5. Panel: Gradyan Normu ve Kararlılık (Gradient Norm)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.plot(epochs, d1["gecmis"]["grad_norm"], label=f"{d1['deney_adi']}", color="#e74c3c", linestyle="--", linewidth=1.8)
        ax5.plot(epochs, d2["gecmis"]["grad_norm"], label=f"{d2['deney_adi']}", color="#3498db", linewidth=2.2)
        ax5.plot(epochs, d3["gecmis"]["grad_norm"], label=f"{d3['deney_adi']}", color="#2ecc71", linewidth=2.2)
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("Ortalama Gradyan Normu (L2)")
        ax5.set_title("5. Gradyan Normu ve Kararlilik", fontweight="bold", color="#27ae60")
        ax5.legend(loc="upper right")

        # -------------------------------------------------------------
        # 6. Panel: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        swot_metin = (
            " OPTIMIZER & SCHEDULER SWOT KARAR MATRISI\n"
            "─────────────────────────────────────────────────\n"
            " [S] GUCLU YONLER (Strengths):\n"
            " • Lion: İkinci moment (v_t) yok, %50 bellek tasarrufu\n"
            " • Sign momentum ile uniform adim buyuklugu\n"
            " • Warmup: Ilk adimlarda gradyan patlamasini onleme\n\n"
            " [W] ZAYIF YONLER (Weaknesses):\n"
            " • Lion icin kucuk LR (3x-10x) ve buyuk batch gereksinimi\n"
            " • AdamW'ye gore hiperparametre hassasiyeti\n\n"
            " [O] FIRSATLAR (Opportunities):\n"
            " • Buyuk Vision Transformer (ViT) egitimlerinde hiz\n"
            " • MLOps hafiza maliyetini ve VRAM tuketimini kisma\n\n"
            " [T] TEHDITLER (Threats):\n"
            " • Yanlis LR seciminde Lion'in osilasyona girmesi"
        )
        ax6.text(
            0.5, 0.5, swot_metin,
            transform=ax6.transAxes,
            fontsize=9.2,
            family="monospace",
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#fef9e7", edgecolor="#f39c12", linewidth=1.8)
        )
        ax6.set_title("6. Optimizasyon SWOT Matrisi", fontweight="bold", color="#d35400")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return cikti_yolu
