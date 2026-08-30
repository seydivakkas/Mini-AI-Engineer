"""
Day 365: 3D-IC Chiplet Architecture & HBM4 Memory Co-Design
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Williams Roofline modelini, LLM çıkarım hızlanmasını,
3D-IC çiplet yerleşimini ve 6-panelli teşhis panosunu çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class ChipletGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü 3D-IC & HBM4 Teşhis Panosu.
    """
    def __init__(self, cikti_dizini: str = None):
        if cikti_dizini is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cikti_dizini = os.path.join(base_dir, "ciktilar")
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI", "Arial"]
        plt.rcParams["axes.edgecolor"] = "#2c3e50"
        plt.rcParams["axes.linewidth"] = 1.2

    def teshis_panelini_ciz(
        self,
        roofline_res: Dict[str, Any],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "chiplet_hbm4_paneli.png"
    ) -> str:
        """
        6 Panelli 3D-IC & HBM4 Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "3D-IC Chiplet Architecture & HBM4 High-Bandwidth Memory Co-Design (Phase 19)",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        intensities = roofline_res["intensities"]
        perf_hbm4 = roofline_res["perf_hbm4"]
        perf_ddr5 = roofline_res["perf_ddr5"]

        # ------------------------------------------------------------------
        # Panel 1: Williams Roofline Modeli (HBM4 8.2 TB/s vs DDR5 128 GB/s)
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(intensities, perf_hbm4, "g-", linewidth=2.5, label="3D-IC + HBM4 (8.19 TB/s)")
        ax1.plot(intensities, perf_ddr5, "r--", linewidth=2.0, label="2D Monolitik + DDR5 (128 GB/s)")
        ax1.axvline(2.0, color="#8e44ad", linestyle=":", label="LLM Token Decoding (2 FLOP/B)")
        ax1.axvline(150.0, color="#d35400", linestyle=":", label="LLM Prefill GEMM (150 FLOP/B)")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_title("1. Williams Roofline Hesaplama ve Bant Genişliği Limiti", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("Operasyonel Yoğunluk (FLOP / Byte - Log)", fontsize=8)
        ax1.set_ylabel("Ulaşılabilir Performans (TFLOPS - Log)", fontsize=8)
        ax1.legend(loc="upper left", fontsize=6)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 2: LLM Bellek-Bağlı Decoding Hızlanması (TFLOPS)
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        dec_ddr = roofline_res["llm_decode_ddr5_tflops"]
        dec_hbm = roofline_res["llm_decode_hbm4_tflops"]
        bars2 = ax2.bar(["2D DDR5 (Bellek Duvarı)", "3D HBM4 (Ultra Bant Genişliği)"], [dec_ddr, dec_hbm], color=["#c0392b", "#27ae60"], width=0.45)
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f} TFLOPS", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax2.set_title(f"2. LLM Token Üretim Hızı ({roofline_res['llm_speedup']:.1f}x Hızlanma)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_ylabel("Decoding Performansı (TFLOPS)", fontsize=8)
        ax2.set_ylim(0, dec_hbm * 1.25)
        ax2.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 3: 3D-IC Çiplet & HBM4 Fiziksel İnterposer Yerleşimi
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        floorplan = np.zeros((10, 10))
        floorplan[3:7, 3:7] = 2.0 # Compute Chiplet (Ortada)
        floorplan[0:2, 3:7] = 1.0 # HBM4 Stack 1
        floorplan[8:10, 3:7] = 1.0 # HBM4 Stack 2
        floorplan[3:7, 0:2] = 1.0 # HBM4 Stack 3
        floorplan[3:7, 8:10] = 1.0 # HBM4 Stack 4
        im3 = ax3.imshow(floorplan, cmap="Accent", origin="lower")
        ax3.set_title("3. CoWoS/3D-IC Paket Yerleşimi (4x HBM4 + GPU)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xticks([])
        ax3.set_yticks([])

        # ------------------------------------------------------------------
        # Panel 4: Dikey TSV Geçiş Gecikmesi ve Parazitik RC
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        layers = ["2D PCB Yolu (Bakır)", "2.5D İnterposer", "3D Dikey TSV"]
        delays = [2500.0, 150.0, 0.75] # ps
        bars4 = ax4.bar(layers, delays, color=["#e74c3c", "#f39c12", "#2ecc71"], width=0.45)
        ax4.set_yscale("log")
        for bar in bars4:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2.0, yval * 1.5, f"{yval:.2f} ps", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax4.set_title("4. Katmanlar Arası Gecikme (Pikosaniye - Log)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_ylabel("Gecikme (ps - Log)", fontsize=8)
        ax4.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 5: Bellek Nesilleri Bant Genişliği Evrimi (TB/s)
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        mems = ["DDR5", "HBM2e", "HBM3e", "HBM4 (Bizim)"]
        bws = [0.128, 1.6, 4.8, 8.192] # TB/s
        bars5 = ax5.bar(mems, bws, color=["#7f8c8d", "#3498db", "#9b59b6", "#27ae60"], width=0.45)
        for bar in bars5:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.2f} TB/s", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax5.set_title("5. Bellek Bant Genişliği Evrimi (TB / saniye)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_ylabel("Bant Genişliği (TB/s)", fontsize=8)
        ax5.set_ylim(0, 10.0)
        ax5.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 6: 3D-IC Chiplet & HBM4 Hazır Bulunurluk Skoru
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        metrics_list = ["HBM4 8.2 TB/s Bant", "3D TSV Dikey Veriyolu", "LLM Decoding Hızlanması", "3D-IC Çiplet Hazırlığı"]
        scores = [
            profiler_metrics.get("hbm4_bandwidth_score", 100.0),
            profiler_metrics.get("tsv_link_score", 99.5),
            profiler_metrics.get("llm_speedup_score", 99.0),
            profiler_metrics.get("chiplet_codesign_readiness", 99.5)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#27ae60", alpha=0.85)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. 3D-IC & HBM4 Eş-Tasarım Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
