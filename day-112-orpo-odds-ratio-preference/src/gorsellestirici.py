"""
ORPO Tercih Hizalaması Teşhis Panosu Görselleştirici Modülü (Day 112).
6 panelli ORPO kayıp eğrileri, Log-Odds oranı gelişimi, PPO vs DPO vs ORPO kıyası ve matematik kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class ORPOGorsellestirici:
    """ORPO analizi için 6 panelli teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        egitim_raporu: Dict[str, List[float]],
        kiyas_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/orpo_alignment_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "Odds Ratio Preference Optimization (ORPO): Tek Aşamalı Monolitik SFT + Tercih Hizalama Paneli",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        epoklar = list(range(1, len(egitim_raporu["toplam_kayiplar"]) + 1))

        # -------------------------------------------------------------
        # PANEL 1: ORPO Kayıp Eğrileri (Total, SFT, OR)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.plot(epoklar, egitim_raporu["toplam_kayiplar"], color="#4e73df", lw=3.0, marker="o", label="Toplam ORPO Kaybı")
        ax1.plot(epoklar, egitim_raporu["kayiplar_sft"], color="#1cc88a", lw=2.0, linestyle="--", label="SFT Kaybı (NLL)")
        ax1.plot(epoklar, egitim_raporu["kayiplar_or"], color="#e74a3b", lw=2.0, linestyle=":", label="Odds Ratio Kaybı (L_OR)")
        ax1.set_title("1. Monolitik ORPO Kayıp Eğrileri (Epoklar Boyunca)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epok")
        ax1.set_ylabel("Kayıp Değeri")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.legend(loc="upper right")

        # -------------------------------------------------------------
        # PANEL 2: Tercih Doğruluğu (% Accuracy)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(epoklar, egitim_raporu["dogruluklar"], color="#28a745", lw=3.0, marker="s", label="Tercih Doğruluğu (%)")
        ax2.axhline(50.0, color="gray", linestyle=":", label="Rastgele Eşik (%50)")
        ax2.axhline(95.0, color="#e74a3b", linestyle="--", label="Hedef Eşik (%95)")
        ax2.set_title("2. Çiftli Sıralama Doğruluğu (Odds w > Odds l)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epok")
        ax2.set_ylabel("Doğruluk (%)")
        ax2.set_ylim(40, 105)
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.legend(loc="lower right")

        # -------------------------------------------------------------
        # PANEL 3: Log-Odds Oranı (Log-Odds Ratio)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(epoklar, egitim_raporu["log_odds_oranlari"], color="#6f42c1", lw=2.5, marker="^", label="Log-Odds Oranı (log OR)")
        ax3.axhline(0.0, color="black", linestyle="--", alpha=0.5)
        ax3.set_title("3. Log-Odds Oranı Gelişimi (log odds(w) - log odds(l))", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Epok")
        ax3.set_ylabel("Log-Odds Oranı")
        ax3.grid(True, linestyle="--", alpha=0.7)
        ax3.legend(loc="lower right")

        # -------------------------------------------------------------
        # PANEL 4: PPO vs DPO vs ORPO Mimari Kıyası
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        yontemler = ["PPO (RLHF)", "DPO", "ORPO (Monolithic)"]
        modeller = [kiyas_raporu["ppo_gpu_model"], kiyas_raporu["dpo_gpu_model"], kiyas_raporu["orpo_gpu_model"]]
        renkler = ["#e74a3b", "#f6c23e", "#1cc88a"]
        barlar = ax4.bar(yontemler, modeller, color=renkler, width=0.55, edgecolor="black")

        for bar, val in zip(barlar, modeller):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val} Model GPU'da\n({100 if val==4 else (50 if val==2 else 25)}% VRAM)",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax4.set_title("4. Hizalama Yöntemleri GPU Model Sayısı ve VRAM Kıyası", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Bellekteki Eşzamanlı Model Sayısı")
        ax4.set_ylim(0, 5)
        ax4.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 5: ORPO Matematik Kartı
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. ORPO Matematik ve Formül Kartı", fontsize=12, fontweight="bold", pad=10)

        formuller = (
            "[i] Odds Ratio Preference Optimization (ORPO):\n"
            "--------------------------------------------------\n"
            "1. Bahis Oranları (Odds Formülasyonu):\n"
            "   odds(y|x) = P(y|x) / ( 1 - P(y|x) )\n\n"
            "2. Log-Odds Oranı (Log-Odds Ratio):\n"
            "   log OR(y_w, y_l) = log odds(y_w) - log odds(y_l)\n\n"
            "3. Monolitik ORPO Kaybı:\n"
            "   L_ORPO = L_SFT(y_w) + lambda * L_OR(y_w, y_l)\n"
            "   L_OR = -log sigma( log OR(y_w, y_l) )\n\n"
            "-> Referans Model YOK! Ayrı SFT aşaması YOK!"
        )

        ax5.text(
            0.05, 0.5, formuller,
            fontsize=8.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: Stajyer Notu & ORPO Karar Sertifikası
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. Stajyer Notu & ORPO Karar Sertifikası", fontsize=12, fontweight="bold", pad=10)

        sertifika = (
            "====================================================\n"
            "            ORPO ALIGNMENT CERTIFICATE              \n"
            "====================================================\n"
            "• Soru: Neden 2 aşamalı (SFT -> DPO) yerine ORPO?   \n"
            "• Cevap: SFT aşaması bazen istenmeyen tokenları da  \n"
            "         öğrenir. ORPO, SFT yaparken eşzamanlı      \n"
            "         olarak kötü yanıtın oranını bastırır!      \n"
            "----------------------------------------------------\n"
            "[ONAYLANDI] Mistral-NeMo & Llama-3-ORPO Standartı!  \n"
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
        print(f"  ✓ ORPO Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
