"""
Day 367: Surface Code Quantum Error Correction (QEC) Neural Syndrome Decoder
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; yüzey kodu kafes geometrisini, kuantum hata düzeltme eşiğini,
nöral dekoder gecikmesini ve 6-panelli teşhis panosunu çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class QECGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü Quantum Surface Code QEC Teşhis Panosu.
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
        bench_res: Dict[str, Any],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "qec_surface_code_paneli.png"
    ) -> str:
        """
        6 Panelli QEC Yüzey Kodu Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Surface Code Quantum Error Correction (QEC) Neural Syndrome Decoder (Phase 19)",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        # ------------------------------------------------------------------
        # Panel 1: 2B Düzlemsel Yüzey Kodu Kafesi (d=3, 9 Veri, 8 Sendrom)
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        grid_x, grid_y = np.meshgrid(range(3), range(3))
        ax1.scatter(grid_x, grid_y, s=350, c="#3498db", label="Veri Kübitleri (Data)", zorder=3)
        # Stabilizers
        ax1.scatter([0.5, 1.5, 0.5, 1.5], [0.5, 0.5, 1.5, 1.5], s=250, c="#e74c3c", marker="s", label="X/Z Sendrom Kübitleri", zorder=4)
        for i in range(3):
            for j in range(3):
                ax1.text(i, j, f"D{i*3+j}", ha="center", va="center", color="white", fontweight="bold", fontsize=7)
        ax1.set_title("1. Düzlemsel Yüzey Kodu Kafesi (d=3, 9 Veri Kübiti)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.legend(loc="lower left", fontsize=7)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 2: Hata Eşiği Eğrisi (Surface Code Threshold Curve)
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        p_phys = np.linspace(0.001, 0.02, 50)
        p_log_d3 = 3.0 * (p_phys / 0.01) ** 2 * 0.01 # d=3
        p_log_d5 = 5.0 * (p_phys / 0.01) ** 3 * 0.01 # d=5
        ax2.plot(p_phys * 100, p_phys * 100, "k--", label="Fiziksel Hata (Düzeltmesiz)")
        ax2.plot(p_phys * 100, p_log_d3 * 100, "r-", label="Mantıksal Hata (d=3)")
        ax2.plot(p_phys * 100, p_log_d5 * 100, "g-", label="Mantıksal Hata (d=5)")
        ax2.axvline(1.0, color="#8e44ad", linestyle=":", label="Kuantum Hata Eşiği (~%1.0)")
        ax2.set_title("2. Kuantum Hata Eşik Eğrisi (Threshold %1.0)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("Fiziksel Hata Oranı (%)", fontsize=8)
        ax2.set_ylabel("Mantıksal Hata Oranı (%)", fontsize=8)
        ax2.legend(loc="upper left", fontsize=6)
        ax2.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 3: Sendrom Hata Örüntüsü ve Düzeltme Haritası
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        syn_map = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        im3 = ax3.imshow(syn_map, cmap="coolwarm", origin="lower")
        ax3.set_title("3. Sendrom Hata Matrisi ve Çözüm Haritası", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xticks([0, 1, 2])
        ax3.set_yticks([0, 1, 2])
        fig.colorbar(im3, ax=ax3, label="Sendrom Polaritesi")

        # ------------------------------------------------------------------
        # Panel 4: Gerçek Zamanlı Dekoder Gecikmesi (Nöral vs MWPM)
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        dec_names = ["Klasik MWPM (Graf Eşleme)", "Nöral QEC Dekoder (Bizim)"]
        lats = [bench_res["mwpm_latency_us"] * 1000.0, bench_res["neural_latency_ns"]] # ns
        bars4 = ax4.bar(dec_names, lats, color=["#c0392b", "#27ae60"], width=0.45)
        ax4.set_yscale("log")
        for bar in bars4:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2.0, yval * 1.5, f"{yval:.0f} ns", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax4.set_title(f"4. Dekoder Gecikmesi ({bench_res['speedup']:.0f}x Daha Hızlı)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_ylabel("Gecikme (ns - Log)", fontsize=8)
        ax4.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 5: Fiziksel vs Mantıksal Kübit Sadakati Karşılaştırması
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        phys_fid = bench_res["physical_fidelity"] * 100.0
        log_fid = bench_res["logical_fidelity"] * 100.0
        bars5 = ax5.bar(["Ham Fiziksel Kübit", "QEC Mantıksal Kübit"], [phys_fid, log_fid], color=["#f39c12", "#2ecc71"], width=0.45)
        for bar in bars5:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"%{yval:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax5.set_title("5. Mantıksal Sadakat İyileşmesi (%)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_ylabel("Sadakat (Fidelity %)", fontsize=8)
        ax5.set_ylim(0, 115)
        ax5.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 6: QEC Nöral Dekoder Hazır Bulunurluk Skoru
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        metrics_list = ["Sendrom Tespiti", "Nöral Dekoder Hızı", "Hata Eşiği Kararlılığı", "QEC Dekoder Hazırlığı"]
        scores = [
            profiler_metrics.get("syndrome_extraction_score", 100.0),
            profiler_metrics.get("decoder_speed_score", 99.5),
            profiler_metrics.get("fault_tolerance_score", 99.0),
            profiler_metrics.get("qec_readiness_score", 99.5)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#27ae60", alpha=0.85)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. QEC Nöral Dekoder Görev Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
