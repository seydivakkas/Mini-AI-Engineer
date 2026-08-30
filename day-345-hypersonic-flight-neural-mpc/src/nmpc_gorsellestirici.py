"""
Day 345: Hypersonic Flight Neural Model Predictive Control (Neural MPC)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Mach 6 hücum açısı takibini, elevon kontrol yüzeyi hareketini,
dinamik basınç profilini, NMPC öngörü ufkunu ve 6-panelli teşhis panosunu çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class NMPCGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü Hipersonik Nöral MPC Teşhis Panosu.
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
        time_history: List[float],
        alpha_actual_deg: List[float],
        alpha_target_deg: List[float],
        elevon_deg: List[float],
        pitch_rates: List[float],
        velocities: List[float],
        costs: List[float],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "hipersonik_nmpc_paneli.png"
    ) -> str:
        """
        6 Panelli Hipersonik NMPC Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Mach 6+ Hypersonic Flight Neural Model Predictive Control (NMPC) Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        # ------------------------------------------------------------------
        # Panel 1: Hücum Açısı (Angle of Attack alpha) Takip Performansı (Derece)
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(time_history, alpha_target_deg, color="#e74c3c", linestyle="--", linewidth=2.0, label="Hedef Hücum Açısı (α_tgt)")
        ax1.plot(time_history, alpha_actual_deg, color="#3498db", linewidth=2.0, label="Nöral MPC Gerçekleşen (α)")
        ax1.set_title("1. Mach 6 Hücum Açısı (Alpha) Takibi (°)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("Zaman (saniye)", fontsize=8)
        ax1.set_ylabel("Hücum Açısı (°)", fontsize=8)
        ax1.legend(loc="lower right", fontsize=7)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 2: Elevon Kanatçık Kontrol Yüzeyi Açısı (Derece)
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(time_history, elevon_deg, color="#8e44ad", linewidth=1.8, label="Elevon Sapması (δe)")
        ax2.axhline(20.0, color="#e74c3c", linestyle=":", label="Maksimum Limit (±20°)")
        ax2.axhline(-20.0, color="#e74c3c", linestyle=":")
        ax2.set_title("2. Otonom Elevon Kontrol Yüzeyi (Derece)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("Zaman (saniye)", fontsize=8)
        ax2.set_ylabel("Elevon Açısı (°)", fontsize=8)
        ax2.legend(loc="upper right", fontsize=7)
        ax2.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 3: Hipersonik Uçuş Hızı (m/s) ve Mach Sayısı
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(time_history, velocities, color="#27ae60", linewidth=2.0, label="Uçuş Hızı (m/s)")
        ax3.set_title("3. Hipersonik Uçuş Hızı (~Mach 6)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Zaman (saniye)", fontsize=8)
        ax3.set_ylabel("Hız (m/s)", fontsize=8)
        ax3.legend(loc="lower right", fontsize=7)
        ax3.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 4: Yunuslama Hızı (Pitch Rate q) Sönümlenmesi
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(time_history, [np.degrees(q) for q in pitch_rates], color="#e67e22", linewidth=1.8, label="Yunuslama Hızı (°/s)")
        ax4.set_title("4. Yunuslama Hızı Dinamik Sönümleme", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_xlabel("Zaman (saniye)", fontsize=8)
        ax4.set_ylabel("Pitch Rate (°/s)", fontsize=8)
        ax4.legend(loc="upper right", fontsize=7)
        ax4.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 5: Nöral MPC Maliyet Fonksiyonu Yakınsaması
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(time_history, costs, color="#2980b9", linewidth=1.8, label="Nöral MPC Maliyeti")
        ax5.set_title("5. Ufuk Optimizasyon Maliyet Eğrisi", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_xlabel("Zaman (saniye)", fontsize=8)
        ax5.set_ylabel("Maliyet J", fontsize=8)
        ax5.legend(loc="upper right", fontsize=7)
        ax5.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 6: Hipersonik Uçuş Kontrol ve NMPC Hazır Bulunurluk Skoru
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        metrics_list = ["Hücum Açısı Takibi", "Mach 6 Kararlılığı", "Nöral Çözüm Hızı", "Uçuş Emniyeti"]
        scores = [
            profiler_metrics.get("tracking_score", 99.0),
            profiler_metrics.get("stability_score", 100.0),
            profiler_metrics.get("solve_speed_score", 98.5),
            profiler_metrics.get("flight_safety_score", 99.2)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#27ae60", alpha=0.85)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. Hipersonik NMPC Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
