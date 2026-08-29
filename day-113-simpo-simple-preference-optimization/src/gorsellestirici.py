"""
SimPO Teşhis Panosu Görselleştirici Modülü (Day 113).
6 panelli SimPO kayıp eğrileri, ödül marjini dinamikleri, hedef marjin ihlal analizi ve 4'lü kıyas matrisi.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class SimPOGorsellestirici:
    """SimPO analizi için 6 panelli teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        egitim_raporu: Dict[str, List[float]],
        kiyas_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/simpo_alignment_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "Simple Preference Optimization (SimPO): Referans Modelsiz & Hedef Marjinli (γ) Hafif Hizalama Paneli",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        epoklar = list(range(1, len(egitim_raporu["kayiplar"]) + 1))

        # -------------------------------------------------------------
        # PANEL 1: SimPO Kayıp Eğrisi
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.plot(epoklar, egitim_raporu["kayiplar"], color="#e74a3b", lw=3.0, marker="o", label="SimPO Kaybı")
        ax1.set_title("1. SimPO Kayıp Eğrisi (Epoklar Boyunca)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epok")
        ax1.set_ylabel("Kayıp (Loss)")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.legend(loc="upper right")

        # -------------------------------------------------------------
        # PANEL 2: Örtük Ödül Gelişimi (Chosen vs Rejected)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(epoklar, egitim_raporu["chosen_odulleri"], color="#28a745", lw=2.5, marker="^", label="r(y_w) - Chosen Ödülü")
        ax2.plot(epoklar, egitim_raporu["rejected_odulleri"], color="#dc3545", lw=2.5, marker="v", linestyle="--", label="r(y_l) - Rejected Ödülü")
        ax2.set_title("2. Örtük Ödüllerin Ayrışması (r = β/|y| * log π)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epok")
        ax2.set_ylabel("Ödül Değeri (Reward)")
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.legend(loc="best")

        # -------------------------------------------------------------
        # PANEL 3: Ödül Marjini (Δr) ve Hedef Marjin (γ=0.5)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(epoklar, egitim_raporu["odul_farklari"], color="#4e73df", lw=3.0, marker="s", label="Ödül Marjini Δr = r_w - r_l")
        ax3.axhline(0.5, color="#f6c23e", lw=2.5, linestyle="--", label="Hedef Marjin (γ = 0.5)")
        ax3.axhline(0.0, color="gray", linestyle=":", alpha=0.7)
        ax3.set_title("3. Ödül Marjini (Δr) ve Hedef Marjin (γ) Aşımı", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Epok")
        ax3.set_ylabel("Marjin Değeri")
        ax3.grid(True, linestyle="--", alpha=0.7)
        ax3.legend(loc="upper left")

        # -------------------------------------------------------------
        # PANEL 4: Marjin İhlal Oranı ve Tercih Doğruluğu
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.plot(epoklar, egitim_raporu["dogruluklar"], color="#1cc88a", lw=3.0, marker="o", label="Tercih Doğruluğu (%)")
        ax4.plot(epoklar, egitim_raporu["marjin_ihlalleri"], color="#fd7e14", lw=2.5, linestyle="--", marker="x", label="Marjin İhlal Oranı (%) [Δr < γ]")
        ax4.axhline(50.0, color="gray", linestyle=":", alpha=0.7)
        ax4.set_title("4. Doğruluk ve Marjin İhlal Oranı (%)", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Epok")
        ax4.set_ylabel("Yüzde (%)")
        ax4.set_ylim(-5, 105)
        ax4.grid(True, linestyle="--", alpha=0.7)
        ax4.legend(loc="center right")

        # -------------------------------------------------------------
        # PANEL 5: 4'lü Hizalama Yöntemi Kıyaslama Tablosu
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. PPO vs DPO vs ORPO vs SimPO Kıyas Matrisi", fontsize=12, fontweight="bold", pad=10)

        sutunlar = ["Yöntem", "Ref Model", "Uzunluk Norm", "Hedef Marjin (γ)", "VRAM"]
        veriler = [
            ["PPO", "Evet (pi_ref)", "Yok", "Yok", "%100 (4 Model)"],
            ["DPO", "Evet (pi_ref)", "Opsiyonel", "Yok", "%50 (2 Model)"],
            ["ORPO", "Hayır (0)", "Var", "Yok", "%25 (1 Model)"],
            ["SimPO", "Hayır (0)", "Doğal (1/|y|)", "Var (γ > 0)", "%25 (1 Model)"],
        ]

        tablo = ax5.table(
            cellText=veriler,
            colLabels=sutunlar,
            loc="center",
            cellLoc="center",
        )
        tablo.auto_set_font_size(False)
        tablo.set_fontsize(9)
        tablo.scale(1.15, 1.8)

        # Tablo başlık renklendirme
        for j in range(len(sutunlar)):
            tablo[(0, j)].set_facecolor("#4e73df")
            tablo[(0, j)].set_text_props(color="white", weight="bold")
        # SimPO satırını vurgula
        for j in range(len(sutunlar)):
            tablo[(4, j)].set_facecolor("#d4edda")
            tablo[(4, j)].set_text_props(weight="bold")

        # -------------------------------------------------------------
        # PANEL 6: SimPO Matematik Kartı & Stajyer Notu
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. SimPO Matematik Kartı & Karar Sertifikası", fontsize=12, fontweight="bold", pad=10)

        formuller = (
            "[i] Simple Preference Optimization (SimPO):\n"
            "--------------------------------------------------\n"
            "1. Örtük Ödül (Implicit Reward):\n"
            "   r(x, y) = (beta / |y|) * log pi_theta(y | x)\n\n"
            "2. Hedef Marjinli Tercih Kaybı:\n"
            "   L_SimPO = -log sigma( r(x, y_w) - r(x, y_l) - gamma )\n"
            "   gamma: Hedef ödül marjini (0.5 - 1.4)\n\n"
            "--------------------------------------------------\n"
            "• Referans Model YOK! (%50 VRAM Tasarrufu)\n"
            "• Doğal Uzunluk Normalizasyonu (Uzunluk yanlılığı 0)\n"
            "• Çıkarım (Inference) olasılıklarıyla %100 uyumlu!\n"
            "=================================================="
        )

        ax6.text(
            0.03, 0.5, formuller,
            fontsize=8.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ SimPO Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
