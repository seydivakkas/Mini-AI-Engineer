"""
Day 348: Degraded Visual Environment (DVE) Sensor Fusion (LiDAR + Radar + FLIR)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; LiDAR, Radar ve FLIR sensörlerinin zorlu koşullardaki hatalarını,
adaptif ağırlık değişimini, 3D engel haritasını ve 6-panelli teşhis panosunu çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class DVEGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü DVE Sensör Füzyon Teşhis Panosu.
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
        true_obstacles: np.ndarray,
        lidar_meas: np.ndarray,
        radar_meas: np.ndarray,
        flir_meas: np.ndarray,
        fused_pos: np.ndarray,
        errors_dict: Dict[str, float],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "dve_sensor_fuzyon_paneli.png"
    ) -> str:
        """
        6 Panelli DVE Sensör Füzyon Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Degraded Visual Environment (DVE) Sensor Fusion (LiDAR + Radar + FLIR) Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        # ------------------------------------------------------------------
        # Panel 1: 3D Gerçek Engeller vs Füzyon Kestirimleri
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.scatter(true_obstacles[:, 0], true_obstacles[:, 1], true_obstacles[:, 2], color="#2c3e50", s=60, marker="o", label="Gerçek Engeller")
        ax1.scatter(fused_pos[:, 0], fused_pos[:, 1], fused_pos[:, 2], color="#27ae60", s=40, marker="^", label="Füzyon Kestirimi")
        ax1.set_title("1. 3D Engel Sahası ve Füzyon Kestirimi", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("X (m)", fontsize=7)
        ax1.set_ylabel("Y (m)", fontsize=7)
        ax1.set_zlabel("Z (m)", fontsize=7)
        ax1.legend(loc="upper right", fontsize=6)

        # ------------------------------------------------------------------
        # Panel 2: Sensör Hata Karşılaştırması (RMSE Metre)
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        sensor_names = ["3D LiDAR\n(Tozda Bozulmuş)", "mmWave Radar\n(Kaba)", "FLIR Termal\n(Kızılötesi)", "Adaptif Füzyon\n(Optimum CI)"]
        err_vals = [
            errors_dict.get("lidar_rmse", 0.85),
            errors_dict.get("radar_rmse", 0.45),
            errors_dict.get("flir_rmse", 0.38),
            errors_dict.get("fused_rmse", 0.18)
        ]
        bars2 = ax2.bar(sensor_names, err_vals, color=["#e74c3c", "#f39c12", "#3498db", "#27ae60"], width=0.55)
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f} m", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax2.set_title("2. Sensör Konumlandırma Hatası (RMSE Metre)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_ylabel("RMSE (Metre)", fontsize=8)
        ax2.set_ylim(0, max(err_vals) * 1.25)
        ax2.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 3: Çevresel Bozulma Katsayısı (Gamma) vs Sensör Gürültüsü
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        gammas = np.linspace(0.0, 1.0, 50)
        lidar_curve = 0.05 * np.exp(3.0 * gammas)
        radar_curve = np.full_like(gammas, 0.45)
        flir_curve = 0.15 + 0.35 * gammas
        fused_curve = np.sqrt(1.0 / (1.0/(lidar_curve**2) + 1.0/(radar_curve**2) + 1.0/(flir_curve**2)))

        ax3.plot(gammas * 100, lidar_curve, "r--", linewidth=1.8, label="LiDAR (Tozda Patlar)")
        ax3.plot(gammas * 100, radar_curve, "y-.", linewidth=1.8, label="Radar (Tozdan Etkilenmez)")
        ax3.plot(gammas * 100, flir_curve, "b:", linewidth=1.8, label="FLIR Termal")
        ax3.plot(gammas * 100, fused_curve, "g-", linewidth=2.2, label="Füzyon Kestirimi")
        ax3.set_title("3. Görüş Bozulması (Brownout) vs Hata", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Görüş Bozulma Oranı (%)", fontsize=8)
        ax3.set_ylabel("Gürültü Standart Sapması σ (m)", fontsize=8)
        ax3.legend(loc="upper left", fontsize=7)
        ax3.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 4: 2D Üstten Görünüm (XY) Sensör Dağılımları
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.scatter(true_obstacles[:, 0], true_obstacles[:, 1], color="#2c3e50", s=80, marker="o", label="Gerçek")
        ax4.scatter(radar_meas[:, 0], radar_meas[:, 1], color="#f39c12", alpha=0.5, s=25, label="Radar")
        ax4.scatter(flir_meas[:, 0], flir_meas[:, 1], color="#3498db", alpha=0.5, s=25, label="FLIR")
        ax4.scatter(fused_pos[:, 0], fused_pos[:, 1], color="#27ae60", s=45, marker="^", label="Füzyon")
        ax4.set_title("4. 2D Yatay Engel Dağılımı ve Füzyon", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_xlabel("X (m)", fontsize=8)
        ax4.set_ylabel("Y (m)", fontsize=8)
        ax4.legend(loc="upper right", fontsize=6)
        ax4.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 5: Emniyetli Helikopter / İHA İniş Bölgesi Kontrolü
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        circle_lz = plt.Circle((0, 0), 10.0, color="#2ecc71", alpha=0.25, label="Güvenli İniş Koridoru (r=10m)")
        ax5.add_patch(circle_lz)
        ax5.scatter(0, 0, color="#27ae60", s=120, marker="H", label="Helipad Merkezi")
        ax5.scatter(fused_pos[:, 0], fused_pos[:, 1], color="#e74c3c", s=40, marker="x", label="Tespit Edilen Engeller")
        ax5.set_xlim(-25, 25)
        ax5.set_ylim(-25, 25)
        ax5.set_title("5. Emniyetli İniş Bölgesi (Safe Landing Zone)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_xlabel("X (m)", fontsize=8)
        ax5.set_ylabel("Y (m)", fontsize=8)
        ax5.legend(loc="upper right", fontsize=6)
        ax5.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 6: DVE Sensör Füzyon Hazır Bulunurluk Skoru
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        metrics_list = ["Toz Penetrasyonu", "Engel Ayrıştırma", "Füzyon Doğruluğu", "DVE Uçuş Güvenliği"]
        scores = [
            profiler_metrics.get("penetration_score", 98.0),
            profiler_metrics.get("resolution_score", 97.5),
            profiler_metrics.get("fusion_accuracy_score", 99.0),
            profiler_metrics.get("dve_safety_score", 98.5)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#27ae60", alpha=0.85)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. DVE Çoklu-Sensör Füzyon Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
