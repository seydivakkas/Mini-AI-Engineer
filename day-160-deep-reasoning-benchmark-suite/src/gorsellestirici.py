"""
FAZ 8 BÜYÜK FİNALİ Teşhis Panosu Görselleştirici Modülü (Day 160 - FAZ 8 BÜYÜK FİNALİ).
6 panelli AIME Kıyası, GPQA/ARC Başarımı, Test-Time Compute Skalalaması, DRI İndeksi, Mimari Harita ve FAZ 8 Mezuniyet Kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class FinalBenchmarkGorsellestirici:
    """FAZ 8 Büyük Final Teşhis Panosu Üreticisi."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        benchmark_sonucu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/deep_reasoning_benchmark_suite_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(23, 13.5))
        fig.suptitle(
            "GÜN 160: FAZ 8 BÜYÜK FİNALİ — AIME, GPQA & ARC-Challenge Kapsamlı Akıl Yürütme Benchmark Paketi",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        modeller = list(benchmark_sonucu["model_sonuclari"].keys())
        kisa_isimler = ["1. Base LLM", "2. CoT", "3. MCTS+PRM", "4. R1-Distill"]

        # -------------------------------------------------------------
        # PANEL 1: AIME 2024 Benchmark Başarımı (Pass@1)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        aime_skorlari = [benchmark_sonucu["model_sonuclari"][m]["aime_pass1"] for m in modeller]
        renkler1 = ["#e74a3b", "#f6c23e", "#4e73df", "#1cc88a"]

        barlar1 = ax1.bar(kisa_isimler, aime_skorlari, color=renkler1, edgecolor="black", width=0.45)
        for bar in barlar1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax1.set_title("1. AIME (Olimpiyat Matematiği) Pass@1", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Doğruluk (%)")
        ax1.set_ylim(0, 110)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: GPQA Diamond & ARC-Challenge Kıyaslaması
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        gpqa_skorlari = [benchmark_sonucu["model_sonuclari"][m]["gpqa_pass1"] for m in modeller]
        arc_skorlari = [benchmark_sonucu["model_sonuclari"][m]["arc_pass1"] for m in modeller]

        x = np.arange(len(kisa_isimler))
        w = 0.35

        b1 = ax2.bar(x - w/2, gpqa_skorlari, w, label="GPQA Diamond (Doktora Fen)", color="#36b9cc", edgecolor="black")
        b2 = ax2.bar(x + w/2, arc_skorlari, w, label="ARC-Challenge (Soyut Mantık)", color="#f6c23e", edgecolor="black")

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"%{h:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax2.set_title("2. GPQA Diamond & ARC-Challenge Başarımları", fontsize=12, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(kisa_isimler, fontsize=9)
        ax2.set_ylabel("Doğruluk (%)")
        ax2.set_ylim(0, 115)
        ax2.legend(loc="upper left")
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Test-Time Compute Skalalaması (Pass@1 vs Pass@16)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        pass1 = [benchmark_sonucu["model_sonuclari"][m]["aime_pass1"] for m in modeller]
        pass16 = [benchmark_sonucu["model_sonuclari"][m]["aime_pass16"] for m in modeller]

        ax3.plot(kisa_isimler, pass1, marker="o", lw=2.2, color="#e74a3b", label="Pass@1 (Tek Deneme)")
        ax3.plot(kisa_isimler, pass16, marker="s", lw=2.5, color="#1cc88a", label="Pass@16 (16x Compute Bütçesi)")

        for x_val, y1, y2 in zip(kisa_isimler, pass1, pass16):
            ax3.text(x_val, y1 - 4.5, f"%{y1:.1f}", ha="center", fontsize=8.5, fontweight="bold", color="#e74a3b")
            ax3.text(x_val, y2 + 2.0, f"%{y2:.1f}", ha="center", fontsize=8.5, fontweight="bold", color="#1cc88a")

        ax3.set_title("3. Test-Time Compute Skalalaması (AIME)", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Doğruluk (%)")
        ax3.set_ylim(0, 115)
        ax3.legend(loc="lower right")
        ax3.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Derin Muhakeme İndeksi (DRI) Genel Skoru
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        dri_skorlari = [benchmark_sonucu["model_sonuclari"][m]["derin_muhakeme_indeksi_dri"] for m in modeller]

        barlar4 = ax4.bar(kisa_isimler, dri_skorlari, color=["#6c757d", "#4e73df", "#36b9cc", "#1cc88a"], edgecolor="black", width=0.45)
        for bar in barlar4:
            h = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}/100", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax4.set_title(f"4. Derin Muhakeme İndeksi (DRI: +{benchmark_sonucu['faz8_toplam_kazanc_puani']} Puan Artış)", fontsize=12, fontweight="bold")
        ax4.set_ylabel("DRI Puanı (100 Üzerinden)")
        ax4.set_ylim(0, 110)
        ax4.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 5: FAZ 8 BÜYÜK FİNAL MİMARİ HARİTASI
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. FAZ 8 Mimari Sentezi (Gün 141 - Gün 160)", fontsize=12, fontweight="bold", pad=10)

        harita_metni = (
            "====================================================\n"
            "        FAZ 8 DEEP REASONING CAPSTONE ARCHITECTURE  \n"
            "====================================================\n"
            "  [System 1 vs System 2] -> [Chain-of-Thought <think>]\n"
            "           │                                        \n"
            "           ▼                                        \n"
            "  [Process Reward Models (PRM)] -> [MCTS Arama Ağacı]\n"
            "           │                                        \n"
            "           ▼                                        \n"
            "  [Lean4 Formal Math & SymPy] -> [Self-Verification]\n"
            "           │                                        \n"
            "           ▼                                        \n"
            "  [Chain-of-Verification CoVe] -> [Dynamic Compute] \n"
            "           │                                        \n"
            "           ▼                                        \n"
            "  [Reasoning Distillation] -> [Causal Inference DAG]\n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, harita_metni,
            fontsize=7.3,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: FAZ 8 MEZUNİYET ÖZET KARTI
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. FAZ 8 MEZUNİYET KARTI (%100 TAMAMLANDI)", fontsize=12, fontweight="bold", pad=10)

        mezuniyet_metni = (
            "====================================================\n"
            "    FAZ 8 GRADUATION SUMMARY (DAYS 141 - 160)       \n"
            "====================================================\n"
            "• Modül Durumu       : [TAMAMLANDI %100] (20/20 Gün)\n"
            "• Başlangıç Seviyesi : Base LLM DRI = 34.8/100\n"
            f"• Zirve Başarımı     : DeepSeek-R1 Distill DRI = {benchmark_sonucu['sampiyon_dri']}/100\n"
            f"• Net Muhakeme Artışı: +{benchmark_sonucu['faz8_toplam_kazanc_puani']} Puan (%155 Sıçrama!)\n"
            "----------------------------------------------------\n"
            "KAZANILAN TEMEL YETKİNLİKLER:\n"
            "  1. Test-Time Compute & MCTS Arama Ağaçları\n"
            "  2. Adım Adım Process Reward Model (PRM) Puanlama\n"
            "  3. Lean4 Biçimsel İspat & Z3/SymPy Sembolik Akıl\n"
            "  4. CoVe Halüsinasyon Önleme & R1 Trace Distillation\n"
            "  5. Causal DAG & Do-Calculus Nedensellik Motoru\n"
            "====================================================\n"
            "  SIRADAKİ FAZ: FAZ 9 (Çok Modlu / Multimodal Modeller)\n"
            "     GÜN 161: LLaVA VLM Mimarisi (ViT + MLP + LLM)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, mezuniyet_metni,
            fontsize=7.3,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ FAZ 8 Büyük Final Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
