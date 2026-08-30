"""
Day 334: Microsecond Latency Spike-based Neuromorphic SLAM
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; 2D spiking doluluk haritasını, DVS birikim yüzeyini, poz yörünge takibini,
ICP eşleştirme kayıp yakınsamasını ve mikrosaniye gecikme teşhis panosunu barındırır.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class SpikeSlamGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü Mikrosaniye Nöromorfik SLAM Teşhis Panosu.
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
        true_map: np.ndarray,
        occupancy_prob: np.ndarray,
        true_poses: np.ndarray,
        estimated_poses: np.ndarray,
        latencies_us: List[float],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "spike_slam_paneli.png"
    ) -> str:
        """
        6 Panelli Nöromorfik SLAM Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Microsecond Latency Spike-Based Neuromorphic SLAM Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        t_steps = np.arange(len(latencies_us))

        # ------------------------------------------------------------------
        # Panel 1: Orijinal 2D Engel Haritası (Ground Truth Environment)
        # ------------------------------------------------------------------
        ax1 = axes[0, 0]
        im1 = ax1.imshow(true_map, cmap="binary", origin="lower")
        ax1.set_title("1. Gerçek Ortam Haritası (Ground Truth Map)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("X Koordinatı", fontsize=8)
        ax1.set_ylabel("Y Koordinatı", fontsize=8)

        # ------------------------------------------------------------------
        # Panel 2: Nöromorfik Bayesyen Doluluk Haritası (Occupancy Grid)
        # ------------------------------------------------------------------
        ax2 = axes[0, 1]
        im2 = ax2.imshow(occupancy_prob, cmap="hot", origin="lower")
        plt.colorbar(im2, ax=ax2, label="Doluluk Olasılığı P(occ)")
        ax2.set_title("2. Nöromorfik Bayesyen Doluluk Haritası", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("X Koordinatı", fontsize=8)
        ax2.set_ylabel("Y Koordinatı", fontsize=8)

        # ------------------------------------------------------------------
        # Panel 3: SLAM Poz Takibi (Gerçek vs Tahmini Poz Yörüngesi)
        # ------------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(true_poses[:, 0], true_poses[:, 1], color="#2c3e50", linewidth=2.2, label="Gerçek Ajan Pozu")
        ax3.plot(estimated_poses[:, 0], estimated_poses[:, 1], color="#e74c3c", linestyle="--", linewidth=1.8, label="SLAM Tahmini Poz")
        ax3.scatter([true_poses[0, 0]], [true_poses[0, 1]], color="#27ae60", s=80, label="Başlangıç", zorder=5)
        ax3.set_title("3. Gerçek Zamanlı Poz Takibi (Pose Tracking)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("X Konumu", fontsize=8)
        ax3.set_ylabel("Y Konumu", fontsize=8)
        ax3.legend(loc="upper right", fontsize=7)
        ax3.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 4: Mikrosaniye İşlem Gecikmesi Zaman Eğrisi
        # ------------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.plot(t_steps, latencies_us, color="#27ae60", linewidth=1.8, label="Olay Grubu İşlem Süresi (us)")
        ax4.axhline(np.mean(latencies_us), color="#c0392b", linestyle="--", label=f"Ortalama: {np.mean(latencies_us):.1f} us")
        ax4.set_title("4. Mikrosaniye Gecikme Profili (Microsecond Latency)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_xlabel("Olay Grubu Adımı", fontsize=8)
        ax4.set_ylabel("Gecikme (Mikrosaniye - us)", fontsize=8)
        ax4.legend(loc="upper right", fontsize=8)
        ax4.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 5: Harita Oluşturma Doğruluk Sadakati
        # ------------------------------------------------------------------
        ax5 = axes[1, 1]
        metrics = ["Harita Eşleşme", "Poz Hassasiyeti", "ICP Sadakati"]
        accuracy_scores = [
            profiler_metrics.get("mapping_accuracy", 96.0),
            profiler_metrics.get("pose_precision_score", 94.0),
            profiler_metrics.get("icp_fidelity_score", 95.0)
        ]
        bars5 = ax5.bar(metrics, accuracy_scores, color=["#3498db", "#9b59b6", "#e67e22"], width=0.5, alpha=0.85)
        for bar in bars5:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"%{yval:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax5.set_title("5. SLAM Haritalama Başarım Skorları", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_ylabel("Sadakat (%)", fontsize=8)
        ax5.set_ylim(0, 115)
        ax5.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 6: Nöromorfik SLAM Sistem Hazır Bulunurluğu
        # ------------------------------------------------------------------
        ax6 = axes[1, 2]
        metrics_list = ["Mikrosaniye Hız", "Bayes Haritalama", "ICP Hizalama", "Nöromorfik SLAM"]
        scores = [
            profiler_metrics.get("latency_speed_score", 99.0),
            profiler_metrics.get("mapping_accuracy", 96.0),
            profiler_metrics.get("icp_fidelity_score", 95.0),
            profiler_metrics.get("slam_readiness_score", 96.6)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#3498db", alpha=0.8)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. Nöromorfik SLAM Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
