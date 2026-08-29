"""
Multimodal Omni Benchmark Teşhis Panosu Görselleştirici Modülü (Day 180 - FAZ 9 BÜYÜK FİNALİ).
6 panelli MME, MMBench CircularEval, MathVista, POPE Halüsinasyon ve FAZ 9 Büyük Final Özet Kartı.
"""

import os
import sys
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class MultimodalBenchmarkGorsellestirici:
    """FAZ 9 Büyük Finali Multimodal Omni Benchmark Teşhis Panosu Üreticisi."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        liderlik_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/multimodal_omni_benchmark_paneli.png",
    ):
        """6 panelli FAZ 9 BÜYÜK FİNAL teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(23, 13.5))
        fig.suptitle(
            "GÜN 180 (FAZ 9 BÜYÜK FİNALİ): Multimodal Omni Benchmark Suite (MME, MMBench & MathVista ile 360° Model Doğrulama)",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        modeller = liderlik_raporu["liderlik_tablosu"]
        model_adlari = [m["model_adi"].split("(")[0].strip() for m in modeller]
        omni_skorlar = [m["omni_score"] for m in modeller]

        # -------------------------------------------------------------
        # PANEL 1: Çok Modlu Model Omni-Score Liderlik Tablosu
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        renkler = ["#4e73df", "#1cc88a", "#36b9cc", "#f6c23e", "#e74a3b", "#6f42c1"]
        bars1 = ax1.bar(model_adlari, omni_skorlar, color=renkler, width=0.55, edgecolor="black", linewidth=0.8)

        ax1.set_title("1. Genel Multimodal Omni-Score Liderlik Tablosu", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Bütünleşik Omni Puanı (%)", fontsize=10)
        ax1.set_ylim(0, 100)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        ax1.tick_params(axis="x", rotation=18)

        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 2: MME Benchmark: Algılama (Perception) vs Biliş (Cognition)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        x = np.arange(len(model_adlari))
        width = 0.35

        p_scores = [m["mme"]["perception"] for m in modeller]
        c_scores = [m["mme"]["cognition"] for m in modeller]

        ax2.bar(x - width/2, p_scores, width, label="Perception (Max 2000 pt)", color="#4e73df", alpha=0.85)
        ax2.bar(x + width/2, c_scores, width, label="Cognition (Max 800 pt)", color="#e74a3b", alpha=0.85)

        ax2.set_title("2. MME Benchmark: Algılama vs Biliş Dağılımı", fontsize=12, fontweight="bold")
        ax2.set_ylabel("MME Puanı", fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_adlari, rotation=18)
        ax2.legend(loc="upper right", frameon=True)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: MMBench CircularEval Sağlamlık & Tutarlılık
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        circ_accs = [m["mmbench"]["circular_acc"] for m in modeller]
        vanilla_accs = [m["mmbench"]["vanilla_acc"] for m in modeller]

        ax3.plot(model_adlari, vanilla_accs, marker="o", linewidth=2.2, color="#36b9cc", label="Standart Vanilla Acc (%)")
        ax3.plot(model_adlari, circ_accs, marker="s", linewidth=2.5, color="#f6c23e", label="CircularEval Acc (% - Sağlam)")
        ax3.fill_between(model_adlari, circ_accs, vanilla_accs, color="#e74a3b", alpha=0.15, label="Pozisyon Önyargı Kaybı")

        ax3.set_title("3. MMBench: CircularEval Sağlamlık & Önyargı", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Doğruluk (%)", fontsize=10)
        ax3.set_ylim(30, 100)
        ax3.tick_params(axis="x", rotation=18)
        ax3.legend(loc="lower left", frameon=True)
        ax3.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: MathVista Görsel Matematik & POPE Halüsinasyon
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        math_scores = [m["mathvista"]["dogruluk"] for m in modeller]
        pope_f1s = [m["pope"]["f1_skoru"] for m in modeller]

        ax4_twin = ax4.twinx()
        bar_math = ax4.bar(x - width/2, math_scores, width, color="#6f42c1", alpha=0.75, label="MathVista Doğruluk (%)")
        line_pope = ax4_twin.plot(model_adlari, pope_f1s, color="#1cc88a", marker="D", linewidth=2.5, label="POPE Anti-Halüsinasyon (F1)")

        ax4.set_title("4. MathVista (Görsel Matematik) & POPE (F1)", fontsize=12, fontweight="bold")
        ax4.set_ylabel("MathVista Doğruluk (%)", color="#6f42c1")
        ax4_twin.set_ylabel("POPE F1 Skoru (%)", color="#1cc88a")
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_adlari, rotation=18)
        ax4.set_ylim(20, 90)
        ax4_twin.set_ylim(70, 100)
        ax4.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 5: Liderlik Tablosu & İcra Logu
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Multimodal Omni Benchmark İcra İzi", fontsize=12, fontweight="bold", pad=10)

        log_lines = [
            "====================================================",
            "      MULTIMODAL OMNI BENCHMARK EVALUATION LOG      ",
            "====================================================",
            f"TEST EDİLEN MODEL: {liderlik_raporu['toplam_model_sayisi']} Model (SOTA + Capstone)",
            f"ZİRVE MODEL      : {liderlik_raporu['en_yuksek_skorlu_model']}",
            "----------------------------------------------------",
            "LİDERLİK TABLOSU (AĞIRLIKLI OMNI SKORU):",
        ]
        for m in modeller:
            log_lines.append(f"#{m['siralama']} {m['model_adi'][:22]:<22} : {m['omni_score']:>5.1f}% | MME:{m['mme']['toplam_puan']:>6.1f} | Math:{m['mathvista']['dogruluk']:>4.1f}%")

        log_lines.extend([
            "----------------------------------------------------",
            "BENCHMARK AĞIRLIK DAĞILIMI:",
            "• MME (%30) + MMBench (%30) + MathVista (%25) + POPE (%15)",
            "===================================================="
        ])

        ax5.text(
            0.02, 0.5, "\n".join(log_lines),
            fontsize=6.8,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: FAZ 9 BÜYÜK FİNALİ ÖZET KARTI
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. FAZ 9 BÜYÜK FİNALİ ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        final_metin = (
            "====================================================\n"
            "   FAZ 9 BÜYÜK FİNALİ: MULTIMODAL FOUNDATIONS       \n"
            "====================================================\n"
            "• Tamamlanan Faz     : FAZ 9 (Gün 161 - Gün 180)\n"
            "• Kapsanan Konular   : VLM (LLaVA, Token Compression),\n"
            "                       GUI Agents, Video LLMs, Whisper,\n"
            "                       Speech-to-Speech, LDM, ControlNet,\n"
            "                       DiT, NeRF, 3DGS ve Omni-Eval.\n"
            "• Başarı Oranı       : 20/20 Gün (%100 Tamamlandı)\n"
            "----------------------------------------------------\n"
            "FAZ 9 TEMEL KAZANIMLAR:\n"
            "  1. Görüntü, video, ses ve 3D'yi LLM'lerle uçtan uca bağlama\n"
            "  2. Difüzyon ve Transformer (DiT) üretici modellerini sıfırdan kurma\n"
            "  3. NeRF ve 3DGS ile hacimsel 3D fotogerçekçi sentez\n"
            "  4. MME, MMBench ve MathVista ile 360° SOTA doğrulama\n"
            "====================================================\n"
            "   SIRADAKİ BÜYÜK FAZ: FAZ 10 (Ultra-MLOps & Triton)\n"
            "   Gün 181: Distributed Data Parallel (DDP)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, final_metin,
            fontsize=7.2,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Multimodal Omni Benchmark Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
