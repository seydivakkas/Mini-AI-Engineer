"""
Day 344: Radiation-Hardened Fault-Tolerant Edge AI Inference with Triple Modular Redundancy (TMR)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; TMR üçlü çekirdek oylamasını, radyasyon SEU bit-flip olaylarını,
hata toleranslı doğruluk karşılaştırmasını ve 6-panelli teşhis panosunu çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class TMRGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü Radyasyona Dayanıklı TMR Teşhis Panosu.
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
        core_a_preds: List[int],
        core_b_preds: List[int],
        core_c_preds: List[int],
        tmr_majority_preds: List[int],
        ground_truth: List[int],
        seu_events: List[bool],
        consensus_ratios: List[float],
        repair_history: List[int],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "radyasyon_tmr_paneli.png"
    ) -> str:
        """
        6 Panelli TMR Radyasyon Dayanıklılığı Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Radiation-Hardened Fault-Tolerant Edge AI with Triple Modular Redundancy (TMR) Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        time_steps = np.arange(len(tmr_majority_preds))

        # ------------------------------------------------------------------
        # Panel 1: Üç Çekirdeğin Paralel Çıkarım Çıktıları ve TMR Oylaması
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(time_steps, core_a_preds, "b.", alpha=0.5, label="Core A Tahmin")
        ax1.plot(time_steps, core_b_preds, "rx", alpha=0.5, label="Core B (Bozulan Çekirdek)")
        ax1.plot(time_steps, core_c_preds, "g+", alpha=0.5, label="Core C Tahmin")
        ax1.plot(time_steps, tmr_majority_preds, color="#2c3e50", linewidth=1.5, linestyle="--", label="TMR Çoğunluk Kararı")
        ax1.set_title("1. Üçlü Çekirdek Çıkarımları ve TMR Konsensüsü", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("Telemetri Örnek Adımı", fontsize=8)
        ax1.set_ylabel("Tahmin Edilen Sınıf", fontsize=8)
        ax1.legend(loc="upper right", fontsize=6)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 2: Radyasyon SEU (Bit-Flip) Olayları ve Şiddeti
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        seu_binary = [1 if e else 0 for e in seu_events]
        ax2.stem(time_steps, seu_binary, linefmt="r-", markerfmt="ro", basefmt="k-")
        ax2.set_title("2. Kozmik Radyasyon SEU Bit-Flip Olayları", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("Zaman Adımı", fontsize=8)
        ax2.set_ylabel("SEU Tetiklendi mi (1/0)", fontsize=8)
        ax2.set_ylim(-0.1, 1.2)
        ax2.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 3: TMR Çoğunluk Oylaması Konsensüs Oranı (2/3 vs 3/3)
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(time_steps, [r * 100.0 for r in consensus_ratios], color="#27ae60", linewidth=2.0, label="TMR Konsensüs Oranı (%)")
        ax3.axhline(66.67, color="#e67e22", linestyle=":", label="2/3 Çoğunluk Eşiği")
        ax3.set_title("3. TMR Çoğunluk Oylaması Güvenilirlik Oranı (%)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Zaman Adımı", fontsize=8)
        ax3.set_ylabel("Konsensüs (%)", fontsize=8)
        ax3.set_ylim(50, 105)
        ax3.legend(loc="lower right", fontsize=7)
        ax3.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 4: Tek Çekirdek vs TMR Hata Toleranslı Doğruluk
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        single_acc = profiler_metrics.get("single_core_accuracy", 75.0)
        tmr_acc = profiler_metrics.get("tmr_accuracy", 100.0)
        
        bars4 = ax4.bar(["Standart Tek Çekirdek (Bozulan)", "TMR 3-Çekirdek + Oylama (Bizim)"], [single_acc, tmr_acc], color=["#e74c3c", "#27ae60"], width=0.5)
        for bar in bars4:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2.0, yval - 12.0, f"%{yval:.1f}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax4.set_title("4. SEU Radyasyon Altında Çıkarım Doğruluğu (%)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_ylabel("Doğruluk (%)", fontsize=8)
        ax4.set_ylim(0, 115)
        ax4.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 5: Kümülatif Otonom Bellek Temizleme & Onarım Sayısı
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(time_steps, repair_history, color="#8e44ad", linewidth=2.0, label="Toplam ECC Onarım Sayısı")
        ax5.set_title("5. Otonom Bellek Scrubbing & Onarım Sayacı", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_xlabel("Zaman Adımı", fontsize=8)
        ax5.set_ylabel("Onarım Adedi", fontsize=8)
        ax5.legend(loc="upper left", fontsize=7)
        ax5.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 6: Radyasyona Dayanıklı AI Sistem Hazır Bulunurluk Skoru
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        metrics_list = ["TMR Doğruluk", "SEU Kurtarma", "ECC Scrubbing", "Uzay Güvenliği"]
        scores = [
            profiler_metrics.get("tmr_accuracy", 100.0),
            profiler_metrics.get("seu_recovery_rate", 100.0),
            profiler_metrics.get("scrubbing_efficiency", 100.0),
            profiler_metrics.get("space_rad_hard_score", 100.0)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#27ae60", alpha=0.85)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. Radyasyona Dayanıklı AI Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
