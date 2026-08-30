"""
Day 357: Radar Micro-Doppler Signature Classification for Micro-UAVs and Ballistic Targets
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; 4 farklı hedef tipinin 2D STFT spektrogramlarını,
sınıflandırma güvenilirlik matrisini ve 6-panelli teşhis panosunu çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class RadarGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü Mikro-Doppler Radar Teşhis Panosu.
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
        analysis_res: Dict[str, Any],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "mikro_doppler_radar_paneli.png"
    ) -> str:
        """
        6 Panelli Radar Mikro-Doppler Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Radar Micro-Doppler Signature Classification (Micro-UAV vs Ballistic) Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        target_res = analysis_res["target_results"]

        # ------------------------------------------------------------------
        # Panel 1: Döner Kanat Döner Pervane Drone (Quadcopter) Spektrogramı
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        d_res = target_res["QUADCOPTER_DRONE"]
        im1 = ax1.pcolormesh(d_res["t"], d_res["f"], d_res["Sxx_dB"], shading='gouraud', cmap='inferno')
        ax1.set_title("1. Quadcopter Mikro İHA (Döner Kanat)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("Zaman (s)", fontsize=8)
        ax1.set_ylabel("Frekans (Hz)", fontsize=8)
        fig.colorbar(im1, ax=ax1, label="dB")

        # ------------------------------------------------------------------
        # Panel 2: Kanat Çırpan Kuş Sürüsü (Bird Flapping) Spektrogramı
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        b_res = target_res["BIRD_FLAPPING"]
        im2 = ax2.pcolormesh(b_res["t"], b_res["f"], b_res["Sxx_dB"], shading='gouraud', cmap='viridis')
        ax2.set_title("2. Kuş / Kanat Çırpma (Bio-Target)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("Zaman (s)", fontsize=8)
        ax2.set_ylabel("Frekans (Hz)", fontsize=8)
        fig.colorbar(im2, ax=ax2, label="dB")

        # ------------------------------------------------------------------
        # Panel 3: Sabit Kanat Kamikaze İHA Spektrogramı
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        u_res = target_res["FIXED_WING_UAV"]
        im3 = ax3.pcolormesh(u_res["t"], u_res["f"], u_res["Sxx_dB"], shading='gouraud', cmap='magma')
        ax3.set_title("3. Sabit Kanat Kamikaze İHA (Piston Motor)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Zaman (s)", fontsize=8)
        ax3.set_ylabel("Frekans (Hz)", fontsize=8)
        fig.colorbar(im3, ax=ax3, label="dB")

        # ------------------------------------------------------------------
        # Panel 4: Balistik Füze Harp Başlığı (Precession Wobble) Spektrogramı
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        w_res = target_res["BALLISTIC_WARHEAD"]
        im4 = ax4.pcolormesh(w_res["t"], w_res["f"], w_res["Sxx_dB"], shading='gouraud', cmap='plasma')
        ax4.set_title("4. Balistik Harp Başlığı (Presesyon/Yalpalama)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_xlabel("Zaman (s)", fontsize=8)
        ax4.set_ylabel("Frekans (Hz)", fontsize=8)
        fig.colorbar(im4, ax=ax4, label="dB")

        # ------------------------------------------------------------------
        # Panel 5: Hedef Sınıflandırma Güven Skorları (%)
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        classes = ["Quadcopter", "Kuş (Bird)", "Sabit Kanat", "Balistik Başlık"]
        confs = [
            d_res["prediction"]["confidence"] * 100.0,
            b_res["prediction"]["confidence"] * 100.0,
            u_res["prediction"]["confidence"] * 100.0,
            w_res["prediction"]["confidence"] * 100.0
        ]
        bars5 = ax5.bar(classes, confs, color=["#27ae60", "#2980b9", "#e67e22", "#c0392b"], width=0.5)
        for bar in bars5:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"%{yval:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax5.set_title("5. Sınıflandırma Model Güven Skoru (%)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_ylabel("Güven (%)", fontsize=8)
        ax5.set_ylim(0, 115)
        ax5.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 6: Radar Mikro-Doppler AI Hazır Bulunurluk Skoru
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        metrics_list = ["İHA Teşhisi", "Kuş Ayrıştırma", "Balistik Teşhis", "Hava Savunma Hazırlığı"]
        scores = [
            profiler_metrics.get("uav_detection_score", 100.0),
            profiler_metrics.get("bird_discrimination_score", 100.0),
            profiler_metrics.get("ballistic_id_score", 100.0),
            profiler_metrics.get("radar_ai_readiness", 100.0)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#27ae60", alpha=0.85)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. Mikro-Doppler Radar AI Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
