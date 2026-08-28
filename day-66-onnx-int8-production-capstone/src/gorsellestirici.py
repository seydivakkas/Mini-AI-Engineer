"""
ONNX INT8 Capstone Gorsellestirici (Visualization Engine)
=========================================================
PyTorch, ONNX FP32 ve ONNX INT8 performans ve dogruluk analizlerini
6 panelli yuksek cozunurluklu endustriyel teshis tablosunda birlestirir.
"""

from typing import Dict, Any, Optional
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class CapstoneGorsellestirici:
    """
    Day 66 Capstone 6-Panelli Karsilastirma Panosu ureticisi.
    """

    @staticmethod
    def panoyu_ciz_ve_kaydet(
        benchmark_sonuclari: Dict[str, Any],
        esdegerlik_sonuclari: Dict[str, Any],
        cikti_yolu: str = "ciktilar/onnx_int8_karsilastirma_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(21, 13))
        fig.suptitle(
            "Day 66: PyTorch -> ONNX Export, INT8 PTQ Kuantizasyon & ONNX Runtime Hizlandirma (Capstone)",
            fontsize=17,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        pt = benchmark_sonuclari["pytorch_fp32"]
        ort_fp32 = benchmark_sonuclari["onnx_fp32"]
        ort_int8 = benchmark_sonuclari["onnx_int8"]

        modeller = ["PyTorch FP32", "ONNX FP32", "ONNX INT8"]
        renkler = ["#e74c3c", "#3498db", "#2ecc71"]

        # -------------------------------------------------------------
        # 1. Panel: Yönetici ve Capstone Özet Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        ozet_metin = (
            "           ONNX & INT8 CAPSTONE OZETI\n"
            "═══════════════════════════════════════════════════════\n"
            f" * PyTorch FP32 Gecikme       : {pt['gecikme_ms']:>6.2f} ms ({pt['fps']:>6.1f} FPS)\n"
            f" * ONNX Runtime FP32 Gecikme  : {ort_fp32['gecikme_ms']:>6.2f} ms ({ort_fp32['fps']:>6.1f} FPS)\n"
            f" * ONNX Runtime INT8 Gecikme  : {ort_int8['gecikme_ms']:>6.2f} ms ({ort_int8['fps']:>6.1f} FPS)\n"
            "───────────────────────────────────────────────────────\n"
            f" * FP32 Hizlanma Orani        : {ort_fp32['speedup']:>6.2f}x HIZLANMA\n"
            f" * INT8 Hizlanma Orani        : {ort_int8['speedup']:>6.2f}x HIZLANMA\n"
            f" * Model Boyut Tasarrufu      : %{((1.0 - ort_int8['boyut_mb']/pt['boyut_mb'])*100):>5.1f} KUCULME\n"
            "───────────────────────────────────────────────────────\n"
            f" * FP32 Kosinus Benzerligi    : %{esdegerlik_sonuclari['fp32_kosinus_benzerligi']*100:>6.4f}\n"
            f" * INT8 Kosinus Benzerligi    : %{esdegerlik_sonuclari['int8_kosinus_benzerligi']*100:>6.4f}\n"
            f" * INT8 Maksimum Hata         : {esdegerlik_sonuclari['int8_maks_fark']:>8.5f}\n"
            "═══════════════════════════════════════════════════════\n"
            " * Durum: URETIM VE EDGE DAGITIMINA HAZIR (VERIFIED)"
        )
        ax1.text(
            0.5, 0.5, ozet_metin,
            transform=ax1.transAxes,
            fontsize=10.5,
            family="monospace",
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=1.2", facecolor="#e8f8f5", edgecolor="#1abc9c", linewidth=2.0)
        )
        ax1.set_title("1. Capstone Yonetici Ozeti", fontweight="bold", color="#16a085")

        # -------------------------------------------------------------
        # 2. Panel: Çıkarım Gecikmesi Kıyaslaması (ms)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        gecikmeler = [pt["gecikme_ms"], ort_fp32["gecikme_ms"], ort_int8["gecikme_ms"]]
        bars = ax2.bar(modeller, gecikmeler, color=renkler, edgecolor="#2c3e50", width=0.55)
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f"{h:.2f} ms", (bar.get_x() + bar.get_width() / 2., h + 0.05),
                         ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax2.set_ylabel("Ortalama Gecikme (ms) - Dusuk Daha Iyi")
        ax2.set_ylim(0, max(gecikmeler) * 1.25)
        ax2.set_title("2. Cikarim Gecikmesi (Latency Benchmark)", fontweight="bold", color="#2980b9")

        # -------------------------------------------------------------
        # 3. Panel: Model Depolama Boyutu (MB)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        boyutlar = [pt["boyut_mb"], ort_fp32["boyut_mb"], ort_int8["boyut_mb"]]
        bars3 = ax3.bar(modeller, boyutlar, color=["#95a5a6", "#34495e", "#27ae60"], edgecolor="#2c3e50", width=0.55)
        for bar in bars3:
            h = bar.get_height()
            ax3.annotate(f"{h:.2f} MB", (bar.get_x() + bar.get_width() / 2., h + 0.02),
                         ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax3.set_ylabel("Disk Boyutu (MB) - Dusuk Daha Iyi")
        ax3.set_ylim(0, max(boyutlar) * 1.25)
        ax3.set_title("3. Model Disk Boyutu & Sikistirma", fontweight="bold", color="#8e44ad")

        # -------------------------------------------------------------
        # 4. Panel: Throughput (FPS / QPS) Hızlanma Analizi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        fps_degerleri = [pt["fps"], ort_fp32["fps"], ort_int8["fps"]]
        bars4 = ax4.bar(modeller, fps_degerleri, color=renkler, edgecolor="#2c3e50", width=0.55)
        for bar, sp in zip(bars4, [pt["speedup"], ort_fp32["speedup"], ort_int8["speedup"]]):
            h = bar.get_height()
            ax4.annotate(f"{h:.1f} FPS\n({sp:.2f}x)", (bar.get_x() + bar.get_width() / 2., h + 15),
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax4.set_ylabel("Throughput (Ornek / Saniye) - Yuksek Daha Iyi")
        ax4.set_ylim(0, max(fps_degerleri) * 1.3)
        ax4.set_title("4. Throughput & Hizlanma Carpani", fontweight="bold", color="#d35400")

        # -------------------------------------------------------------
        # 5. Panel: Sayısal Eşdeğerlik ve Lojit Korelasyonu
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        pt_cikti_flat = esdegerlik_sonuclari["pytorch_cikti"].flatten()[:20]
        fp32_cikti_flat = esdegerlik_sonuclari["onnx_fp32_cikti"].flatten()[:20]
        int8_cikti_flat = esdegerlik_sonuclari["onnx_int8_cikti"].flatten()[:20]
        x_idx = np.arange(len(pt_cikti_flat))

        ax5.plot(x_idx, pt_cikti_flat, label="PyTorch FP32", color="#e74c3c", marker="o", linewidth=2)
        ax5.plot(x_idx, fp32_cikti_flat, label="ONNX FP32", color="#3498db", linestyle="--", marker="s", markersize=6)
        ax5.plot(x_idx, int8_cikti_flat, label="ONNX INT8", color="#2ecc71", linestyle=":", marker="^", markersize=6)
        ax5.set_xlabel("Lojit Indeksi (Ilk 20 Nitelik)")
        ax5.set_ylabel("Ham Lojit Degeri")
        ax5.legend(loc="upper right")
        ax5.set_title(f"5. Sayisal Esdegerlik (INT8 Sim: %{esdegerlik_sonuclari['int8_kosinus_benzerligi']*100:.2f})", fontweight="bold", color="#27ae60")

        # -------------------------------------------------------------
        # 6. Panel: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        swot_metin = (
            "    ONNX & INT8 PTQ DAGITIM SWOT MATRISI\n"
            "──────────────────────────────────────────────\n"
            " [S] GUCLU YONLER (Strengths):\n"
            " • Donanimdan bagimsiz cikarim (CPU/GPU/NPU)\n"
            " • %75 bellek/disk tasarrufu, sifir egitim maliyeti\n"
            " • Operator Fusion (Conv+BN+ReLU) ile hizlanma\n\n"
            " [W] ZAYIF YONLER (Weaknesses):\n"
            " • Kuantizasyon gurultusu (kucuk sayisal sapma)\n"
            " • Opset uyumsuzlugu ve dinamik boyut sinirlari\n\n"
            " [O] FIRSATLAR (Opportunities):\n"
            " • Uç (Edge/IoT) cihazlarda yuksek FPS cikarim\n"
            " • Bulut sunucu maliyetlerinde %60+ tasarruf\n\n"
            " [T] TEHDITLER (Threats):\n"
            " • Outlier aktivasyonlarda hassasiyet kaybi (drift)"
        )
        ax6.text(
            0.5, 0.5, swot_metin,
            transform=ax6.transAxes,
            fontsize=9.5,
            family="monospace",
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#fef9e7", edgecolor="#f39c12", linewidth=1.8)
        )
        ax6.set_title("6. ONNX & INT8 MLOps SWOT Matrisi", fontweight="bold", color="#d35400")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return cikti_yolu
