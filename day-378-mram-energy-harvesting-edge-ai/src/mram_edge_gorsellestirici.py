"""
Day 378: Energy-Harvesting STT-MRAM Ultra-Low-Power Edge AI Accelerator
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; ortam enerjisi hasadı profilini, kapasitör voltajını, kesintili
durum makinesini, STT-MRAM TMR direncini ve 6-panelli teşhis panosunu çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class MRAMEdgeGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü Energy-Harvesting MRAM Edge AI Teşhis Panosu.
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
        dosya_adi: str = "mram_edge_ai_paneli.png"
    ) -> str:
        """
        6 Panelli MRAM Edge AI Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Energy-Harvesting STT-MRAM Ultra-Low-Power Edge AI Accelerator (Phase 19)",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        t_axis = bench_res["t_axis"]
        p_harv = bench_res["p_harvest_curve"]
        v_cap = bench_res["v_cap_history"]

        # ------------------------------------------------------------------
        # Panel 1: Ortam Enerjisi Hasat Güç Profili (Ambient Power)
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(t_axis, p_harv, color="#f39c12", linewidth=1.8, label="Hasat Edilen Güç ($P_{harv}$)")
        ax1.axvspan(100, 160, color="#e74c3c", alpha=0.2, label="1. Güç Kesintisi (Brownout)")
        ax1.axvspan(220, 250, color="#e74c3c", alpha=0.2, label="2. Güç Kesintisi")
        ax1.set_title("1. Ortam Enerjisi Hasat Profili (Solar/RF/Piezo)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("Zaman (ms)", fontsize=8)
        ax1.set_ylabel("Güç ($\\mu$W)", fontsize=8)
        ax1.legend(loc="upper right", fontsize=7.5)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 2: Süper-Kapasitör Voltaj Dinamiği ve Eşikler
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(t_axis, v_cap, color="#2980b9", linewidth=2.0, label="Kapasitör Voltajı ($V_{cap}$)")
        ax2.axhline(3.3, color="#27ae60", linestyle="--", label="$V_{max} = 3.3\\text{V}$")
        ax2.axhline(2.0, color="#e67e22", linestyle=":", label="$V_{brownout} = 2.0\\text{V}$ (Checkpoint)")
        ax2.axhline(1.8, color="#c0392b", linestyle="--", label="$V_{min} = 1.8\\text{V}$ (Kapanma)")
        ax2.set_title("2. Süper-Kapasitör Voltaj Dinamiği ve Eşikler", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("Zaman (ms)", fontsize=8)
        ax2.set_ylabel("Voltaj (V)", fontsize=8)
        ax2.legend(loc="lower right", fontsize=7)
        ax2.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 3: Kesintili Hesaplama (Intermittent) Durum Haritası
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        state_map = {"INFERENCE": 3, "CHECKPOINTING": 2, "CHARGING": 1, "SLEEP_ZERO_POWER": 0}
        numeric_states = [state_map.get(s, 0) for s in bench_res["state_history"]]
        ax3.step(t_axis, numeric_states, color="#8e44ad", linewidth=1.5, where="post")
        ax3.set_yticks([0, 1, 2, 3])
        ax3.set_yticklabels(["Uyku (0W)", "Şarj", "Checkpoint", "AI Çıkarım"], fontsize=7.5)
        ax3.set_title("3. Kesintili Hesaplama Güç Durum Akışı", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Zaman (ms)", fontsize=8)
        ax3.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 4: STT-MRAM Manyetik Tünel Eklemi (MTJ) TMR Dirençleri
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        bars4 = ax4.bar(
            ["Paralel ($R_P$)", "Anti-Paralel ($R_{AP}$)"],
            [1.0, 2.5],
            color=["#16a085", "#d35400"],
            width=0.45
        )
        for bar in bars4:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.08, f"{yval:.1f} k$\\Omega$", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax4.set_title(f"4. STT-MRAM MTJ Direnç Durumları (TMR: %{bench_res['tmr_ratio']:.1f})", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_ylabel("Direnç (k$\\Omega$)", fontsize=8)
        ax4.set_ylim(0, 3.2)
        ax4.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 5: Statik Sızıntı Enerjisi Kaybı (SRAM vs MRAM)
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        bars5 = ax5.bar(
            ["Geleneksel SRAM", "Uçucu Olmayan STT-MRAM"],
            [bench_res["sram_leakage_uj"], bench_res["mram_leakage_uj"]],
            color=["#e74c3c", "#27ae60"],
            width=0.45
        )
        for bar in bars5:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f} $\\mu$J", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax5.set_title("5. Standby Statik Sızıntı Enerjisi (Sıfır Güç)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_ylabel("Sızıntı Enerjisi ($\\mu$J)", fontsize=8)
        ax5.set_ylim(0, max(0.01, bench_res["sram_leakage_uj"] * 1.35))
        ax5.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 6: Energy-Harvesting Edge AI Hazır Bulunurluk Skoru
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        metrics_list = ["Sıfır-Sızıntı Tasarrufu", "Kesintisiz İlerleme", "TMR Kararlılığı", "Edge AI Hazırlığı"]
        scores = [
            profiler_metrics.get("leakage_savings_score", 100.0),
            profiler_metrics.get("forward_progress_score", 100.0),
            profiler_metrics.get("tmr_stability_score", 99.0),
            profiler_metrics.get("edge_ai_readiness_score", 99.7)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#27ae60", alpha=0.85)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. Energy-Harvesting Edge AI Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
