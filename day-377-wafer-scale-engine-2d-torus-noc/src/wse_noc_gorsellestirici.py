"""
Day 377: Wafer-Scale Engine (WSE) 2D-Torus Network-on-Chip (NoC) & Fault Tolerance
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; 2D Wafer kusur haritasını, 2D-Torus yönlendirme yörüngelerini,
gecikme dağılımını ve 6-panelli WSE NoC teşhis panosunu çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class WSENoCGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü WSE 2D-Torus NoC Teşhis Panosu.
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
        dosya_adi: str = "wse_2d_torus_noc_paneli.png"
    ) -> str:
        """
        6 Panelli WSE 2D-Torus Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Wafer-Scale Engine (WSE) 2D-Torus Network-on-Chip & Fault-Tolerant Routing (Phase 19)",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        width = bench_res["width"]
        height = bench_res["height"]
        defect_map = bench_res["defect_map"]
        h_res = bench_res["healthy"]
        f_res = bench_res["faulty"]

        # ------------------------------------------------------------------
        # Panel 1: Wafer Silikon Çekirdek Izgarası ve Kusur Haritası
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        grid_vis = np.zeros((width, height))
        grid_vis[defect_map] = 1.0  # 1 = Defect (Kırmızı)
        
        cax1 = ax1.imshow(grid_vis.T, cmap="Greens", origin="lower", alpha=0.7)
        # Kusurlu noktaları kırmızı X ile işaretle
        dy, dx = np.where(defect_map.T)
        ax1.scatter(dx, dy, color="#e74c3c", marker="X", s=80, label="Silikon Kusuru (Bypass)")
        ax1.set_title("1. Wafer Çekirdek Matrisi ve Kusur Haritası (%5 Hata)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("X Izgara Konumu (PE)", fontsize=8)
        ax1.set_ylabel("Y Izgara Konumu (PE)", fontsize=8)
        ax1.legend(loc="upper right", fontsize=7.5)
        ax1.grid(True, linestyle=":", alpha=0.4)

        # ------------------------------------------------------------------
        # Panel 2: 2D-Torus XY DOR Yönlendirme ve Baypas Yörüngesi
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.imshow(grid_vis.T, cmap="Greens", origin="lower", alpha=0.3)
        # Örnek teslim edilen bir paketin yolunu çiz
        if f_res.get("delivered_packets"):
            sample_pkt = f_res["delivered_packets"][0]
            path = sample_pkt.route_path
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax2.plot(px, py, "b-o", linewidth=2, markersize=5, label=f"Paket #{sample_pkt.packet_id} Yörüngesi")
            ax2.scatter([px[0]], [py[0]], color="#27ae60", marker="s", s=100, label="Kaynak (Src)", zorder=5)
            ax2.scatter([px[-1]], [py[-1]], color="#8e44ad", marker="*", s=150, label="Hedef (Dst)", zorder=5)
        ax2.set_title("2. 2D-Torus Boyut-Sıralı Baypas Yolu (DOR)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("X Izgara", fontsize=8)
        ax2.set_ylabel("Y Izgara", fontsize=8)
        ax2.legend(loc="upper right", fontsize=7)
        ax2.grid(True, linestyle=":", alpha=0.4)

        # ------------------------------------------------------------------
        # Panel 3: Atlama Sayısı (Hop Count) Dağılımı (Sağlıklı vs Kusurlu)
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        h_hops = [p.hop_count for p in h_res["delivered_packets"]]
        f_hops = [p.hop_count for p in f_res["delivered_packets"]]
        ax3.hist(h_hops, bins=np.arange(0, 20, 1), alpha=0.6, color="#2980b9", label=f"Kusursuz (Ort: {h_res['avg_hops']:.1f})", edgecolor="black")
        ax3.hist(f_hops, bins=np.arange(0, 20, 1), alpha=0.6, color="#e67e22", label=f"%5 Kusurlu (Ort: {f_res['avg_hops']:.1f})", edgecolor="black")
        ax3.set_title("3. Flit Atlama Sayısı (Hop Count) Dağılımı", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Atlama Sayısı (Hops)", fontsize=8)
        ax3.set_ylabel("Paket Adedi", fontsize=8)
        ax3.legend(loc="upper right", fontsize=7.5)
        ax3.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 4: Paket Teslim Başarısı (%100 Sıfır Paket Kaybı)
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        bars4 = ax4.bar(
            ["Kusursuz Wafer", "%5 Kusurlu Wafer (Baypas)"],
            [h_res["delivery_rate"], f_res["delivery_rate"]],
            color=["#27ae60", "#2ecc71"],
            width=0.45
        )
        for bar in bars4:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2.0, yval - 10.0, f"%{yval:.1f}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax4.set_title("4. Paket Teslim Başarısı (Sıfır Paket Kaybı)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_ylabel("Teslim Oranı (%)", fontsize=8)
        ax4.set_ylim(0, 115)
        ax4.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 5: Toplam Wafer Bisection Bant Genişliği (PB/s)
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        categories = ["Standart 8xGPU NVLink", "Cerebras WSE-2 Tarzı", "Simüle Edilen 2D-Torus"]
        bws = [0.0072, 0.220, bench_res["bisection_bw_pbps"]]
        bars5 = ax5.bar(categories, bws, color=["#7f8c8d", "#9b59b6", "#e74c3c"], width=0.45)
        for bar in bars5:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f} PB/s", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax5.set_title("5. Bisection Bant Genişliği Karşılaştırması", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_ylabel("Bant Genişliği (PetaBytes/sec)", fontsize=8)
        ax5.set_ylim(0, max(bws) * 1.35)
        ax5.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 6: WSE-3 Ultra-Scale Hazır Bulunurluk Skoru
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        metrics_list = ["Teslimat Güvenilirliği", "Kusur Toleransı", "Toroidal Verimlilik", "WSE NoC Hazırlığı"]
        scores = [
            profiler_metrics.get("delivery_score", 100.0),
            profiler_metrics.get("fault_tolerance_score", 100.0),
            profiler_metrics.get("torus_efficiency_score", 95.0),
            profiler_metrics.get("wse_readiness_score", 98.3)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#27ae60", alpha=0.85)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. Wafer-Scale AI NoC Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
