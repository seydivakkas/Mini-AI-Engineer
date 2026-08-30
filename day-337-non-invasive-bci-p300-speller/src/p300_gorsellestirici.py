"""
Day 337: Non-Invasive BCI P300 Speller & Error-Related Potential (ErrP) Real-Time Correction
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; 6x6 BCI speller matrisini, P300 ERP dalga formunu, ErrP hata dalgasını,
ham vs ErrP düzeltmeli yazma doğruluklarını ve ITR performans panosunu barındırır.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class P300Gorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü BCI P300 Speller & ErrP Düzeltme Teşhis Panosu.
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
        time_vec: np.ndarray,
        target_erp: np.ndarray,
        nontarget_erp: np.ndarray,
        errp_wave: np.ndarray,
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "p300_speller_paneli.png"
    ) -> str:
        """
        6 Panelli BCI P300 Speller Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Non-Invasive BCI P300 Speller & Error-Related Potential (ErrP) Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        # ------------------------------------------------------------------
        # Panel 1: 6x6 BCI Speller Matrisi Görünümü
        # ------------------------------------------------------------------
        ax1 = axes[0, 0]
        grid = [
            ['A', 'B', 'C', 'D', 'E', 'F'],
            ['G', 'H', 'I', 'J', 'K', 'L'],
            ['M', 'N', 'O', 'P', 'Q', 'R'],
            ['S', 'T', 'U', 'V', 'W', 'X'],
            ['Y', 'Z', '0', '1', '2', '3'],
            ['4', '5', '6', '7', '8', '9']
        ]
        ax1.set_xlim(-0.5, 5.5)
        ax1.set_ylim(5.5, -0.5)
        ax1.set_xticks(range(6))
        ax1.set_yticks(range(6))
        ax1.set_xticklabels([f"C{i+1}" for i in range(6)])
        ax1.set_yticklabels([f"R{i+1}" for i in range(6)])
        
        for r in range(6):
            for c in range(6):
                bg_color = "#f1c40f" if (r == 1 or c == 2) else "#ecf0f1"
                rect = plt.Rectangle((c-0.45, r-0.45), 0.9, 0.9, facecolor=bg_color, edgecolor="#2c3e50", lw=1.5)
                ax1.add_patch(rect)
                ax1.text(c, r, grid[r][c], ha="center", va="center", fontsize=11, fontweight="bold", color="#2c3e50")

        ax1.set_title("1. 6x6 BCI Speller Matrisi (Çakma Vurgusu)", fontsize=10, fontweight="bold", color="#2c3e50")

        # ------------------------------------------------------------------
        # Panel 2: Multi-Kanal EEG P300 ERP Dalgası (Target vs Non-Target)
        # ------------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(time_vec * 1000, target_erp, color="#e74c3c", linewidth=2.2, label="Hedef (Target P300 Wave)")
        ax2.plot(time_vec * 1000, nontarget_erp, color="#95a5a6", linewidth=1.5, linestyle="--", label="Hedef Dışı (Non-Target)")
        ax2.axvline(300, color="#27ae60", linestyle=":", label="P300 Tepe Noktası (300 ms)")
        ax2.set_title("2. EEG Olaya İlişkin Potansiyel (P300 ERP)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("Zaman (ms)", fontsize=8)
        ax2.set_ylabel("Genlik (µV)", fontsize=8)
        ax2.legend(loc="upper left", fontsize=7)
        ax2.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 3: Hata Potansiyeli (ErrP N250 / P450 Dalgası)
        # ------------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(time_vec * 1000, errp_wave, color="#8e44ad", linewidth=2.2, label="ErrP Sinyali (N250 / P450)")
        ax3.axvline(250, color="#e74c3c", linestyle=":", label="N250 Negatif Tepe (250 ms)")
        ax3.axhline(-3.5, color="#2c3e50", linestyle="--", label="ErrP Tespiti Eşiği (-3.5 µV)")
        ax3.set_title("3. Hata Potansiyeli (ErrP - Error Potential)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Zaman (ms)", fontsize=8)
        ax3.set_ylabel("Genlik (µV)", fontsize=8)
        ax3.legend(loc="lower right", fontsize=7)
        ax3.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 4: Ham Doğruluk vs ErrP Düzeltmeli BCI Doğruluğu
        # ------------------------------------------------------------------
        ax4 = axes[1, 0]
        methods = ["Ham BCI", "ErrP Düzeltmeli BCI"]
        accuracies = [profiler_metrics.get("raw_accuracy", 72.0), profiler_metrics.get("corrected_accuracy", 96.0)]
        bars4 = ax4.bar(methods, accuracies, color=["#e67e22", "#27ae60"], width=0.4, alpha=0.85)
        for bar in bars4:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"%{yval:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax4.set_title("4. BCI Karakter Yazma Doğruluğu (%)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_ylabel("Doğruluk (%)", fontsize=8)
        ax4.set_ylim(0, 115)
        ax4.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 5: Bilgi Transfer Hızı (ITR bits/min) vs Deneme Sayısı
        # ------------------------------------------------------------------
        ax5 = axes[1, 1]
        trials = np.arange(1, 11)
        itrs = profiler_metrics.get("itr_history", [15, 22, 28, 35, 42, 48, 52, 55, 58, 62])
        ax5.plot(trials, itrs, color="#3498db", marker="o", linewidth=2.0, label="ITR (bits/min)")
        ax5.set_title("5. Bilgi Transfer Hızı (ITR bits/min)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_xlabel("Yazma Deneme Sayısı", fontsize=8)
        ax5.set_ylabel("ITR (bits/min)", fontsize=8)
        ax5.legend(loc="lower right", fontsize=8)
        ax5.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 6: BCI Speller Sistem Hazır Bulunurluk Skoru
        # ------------------------------------------------------------------
        ax6 = axes[1, 2]
        metrics_list = ["P300 Tespiti", "ErrP Düzeltme", "ITR Performansı", "BCI Speller Sistemi"]
        scores = [
            profiler_metrics.get("p300_detection_score", 95.0),
            profiler_metrics.get("errp_correction_score", 96.0),
            profiler_metrics.get("itr_score", 94.0),
            profiler_metrics.get("bci_readiness_score", 95.0)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#3498db", alpha=0.8)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. BCI P300 Speller Sistem Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
