"""
Reasoning Trace Distillation Teşhis Panosu Görselleştirici Modülü (Day 158 - Faz 8).
6 panelli Benchmark Kıyası, SFT Kayıp Eğrisi, Doğruluk Sıçraması, Düşünce İzi Örneği, Damıtma Şeması ve Özet Kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class DamitmaGorsellestirici:
    """Reasoning Trace Distillation teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        egitim_sonucu: Dict[str, Any],
        ornek_iz: Dict[str, Any],
        kayit_yolu: str = "ciktilar/reasoning_trace_distillation_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle(
            "GÜN 158: Büyük Akıl Yürüten Modelin (DeepSeek-R1) Düşünce İncilerini Küçük Modele Damıtma (Reasoning Trace Distillation)",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Benchmark Doğruluk Kıyası (MATH & GSM8K)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        modeller = ["Öğretmen (R1 671B)", "Ham Öğrenci (1.5B)", "Damıtılmış (1.5B R1)"]
        math_skorlari = [92.4, 28.6, 84.2]
        gsm8k_skorlari = [96.8, 54.2, 89.6]

        x = np.arange(len(modeller))
        width = 0.35

        b1 = ax1.bar(x - width/2, math_skorlari, width, label="MATH Benchmark (%)", color="#4e73df", edgecolor="black")
        b2 = ax1.bar(x + width/2, gsm8k_skorlari, width, label="GSM8K Benchmark (%)", color="#1cc88a", edgecolor="black")

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"%{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax1.set_title("1. Model Akıl Yürütme Benchmark Başarımları", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(modeller, fontsize=9.5)
        ax1.set_ylabel("Doğruluk Oranı (%)")
        ax1.set_ylim(0, 115)
        ax1.legend(loc="upper left")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: SFT Damıtma Kayıp Eğrisi (Loss)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(egitim_sonucu["adimlar"], egitim_sonucu["kayiplar"], marker="s", color="#e74a3b", lw=2.5, label="SFT Distillation Loss")
        for x_val, y_val in zip(egitim_sonucu["adimlar"], egitim_sonucu["kayiplar"]):
            ax2.text(x_val, y_val + 0.08, f"{y_val:.2f}", ha="center", fontsize=9, fontweight="bold")

        ax2.set_title("2. SFT Düşünce İzi Damıtma Kaybı (Loss)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Eğitim Adımları (Steps)")
        ax2.set_ylabel("Cross-Entropy Loss")
        ax2.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Küçük Öğrenci Doğruluk İlerlemesi (%)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(egitim_sonucu["adimlar"], egitim_sonucu["ogrenci_dogruluk_egrisi"], marker="o", color="#36b9cc", lw=2.5, label="1.5B MATH Doğruluğu (%)")
        for x_val, y_val in zip(egitim_sonucu["adimlar"], egitim_sonucu["ogrenci_dogruluk_egrisi"]):
            ax3.text(x_val, y_val + 1.8, f"%{y_val:.1f}", ha="center", fontsize=9, fontweight="bold")

        ax3.set_title(f"3. 1.5B Model Doğruluk Sıçraması (+%{egitim_sonucu['performans_kazanci_yuzde']})", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Eğitim Adımları")
        ax3.set_ylabel("Doğruluk (%)")
        ax3.set_ylim(0, 105)
        ax3.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Örnek Damıtılmış Düşünce İzi (<think>...)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. Damıtılan Düşünce İzi Örneği (<think>...)", fontsize=12, fontweight="bold", pad=10)

        iz_metni = (
            "====================================================\n"
            "         CURATED TEACHER REASONING TRACE            \n"
            "====================================================\n"
            f"SORU: {ornek_iz['soru']}\n"
            "----------------------------------------------------\n"
            f"{ornek_iz['ham_iz']}\n"
            "===================================================="
        )

        ax4.text(
            0.02, 0.5, iz_metni,
            fontsize=7.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: DeepSeek-R1 Damıtma Mimarisi
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. DeepSeek-R1 Distillation Mimarisi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "      REASONING TRACE DISTILLATION PIPELINE         \n"
            "====================================================\n"
            "  [DeepSeek-R1 671B Öğretmen (Saf RL ile Eğitilmiş)] \n"
            "           │                                        \n"
            "           ▼  800k+ Düşünce İzi (<think>...</think>)\n"
            "  [Düşünce İzi Kalite & Doğruluk Süzgeci]           \n"
            "    - Hatalı / döngülü zincirleri temizle           \n"
            "    - Refleksif 'Aha moment' adımlarını sakla       \n"
            "           │                                        \n"
            "           ▼  Kürate Edilmiş SFT Veri Seti           \n"
            "  [Küçük Öğrenci Model (Qwen-1.5B / 7B / 14B)]      \n"
            "           │                                        \n"
            "           ▼                                        \n"
            "  [R1-Distill Küçük Model (%91 Öğretmen Seviyesi)]  \n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, sema_metni,
            fontsize=7.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 158 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 158 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   DAY 158 SUMMARY: REASONING TRACE DISTILLATION    \n"
            "====================================================\n"
            f"• Öğretmen Model         : DeepSeek-R1 (671B MoE)\n"
            f"• Öğrenci Model          : Qwen-1.5B (Kompakt Uç Cihaz)\n"
            f"• Ham Öğrenci MATH       : %28.6\n"
            f"• Damıtılmış Öğrenci MATH: %84.2 (+%{egitim_sonucu['performans_kazanci_yuzde']} Sıçrama)\n"
            f"• Öğretmeni Yakalama     : %{egitim_sonucu['ogretmen_yakalama_orani']} Seviyesinde\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. Küçük modelin saf RL olmadan akıl yürütme kazanması\n"
            "  2. Açık düşünce (<think>) etiketleriyle SFT eğitimi\n"
            "  3. Döngülü/çöp düşünce zincirlerinin filtrelenmesi\n"
            "  4. 400x daha ucuz çıkarımla olimpik matematik çözümü\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 159 (Causal DAG & Do-Calculus Engine)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, ozet_metin,
            fontsize=7.8,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Reasoning Trace Distillation Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
