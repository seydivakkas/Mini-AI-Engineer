"""
Day 347: Decentralized Drone Swarm Flocking with Graph Neural Networks (GNN)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; 3D İHA sürü uçuş izlerini, sürü içi minimum mesafeyi,
hız mutabakatını (consensus), dinamik graf bağlantılarını ve 6-panelli teşhis panosunu çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class FlockingGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü GNN İHA Sürü Flocking Teşhis Panosu.
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
        drone_trajectories: np.ndarray,
        min_distances_m: List[float],
        velocity_variances: List[float],
        target_waypoint: np.ndarray,
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "iha_flocking_paneli.png"
    ) -> str:
        """
        6 Panelli İHA Sürü Flocking Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Decentralized Drone Swarm Flocking with Graph Neural Networks (GNN) Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        time_steps = np.arange(len(min_distances_m))
        num_drones = drone_trajectories.shape[1]

        # ------------------------------------------------------------------
        # Panel 1: 3D İHA Sürü Uçuş İzleri ve Formasyon Yolu
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        for i in range(num_drones):
            traj = drone_trajectories[:, i, :]
            ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.6, linewidth=1.2)
            ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], s=25)
        ax1.scatter(target_waypoint[0], target_waypoint[1], target_waypoint[2], color="#e74c3c", s=150, marker="*", label="Görev Hedefi (Waypoint)")
        ax1.set_title(f"1. 3D İHA Sürüsü ({num_drones} İHA) Flocking İzleri", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("X (m)", fontsize=7)
        ax1.set_ylabel("Y (m)", fontsize=7)
        ax1.set_zlabel("Z (m)", fontsize=7)
        ax1.legend(loc="upper right", fontsize=6)

        # ------------------------------------------------------------------
        # Panel 2: İHA'lar Arası Minimum Mesafe ve Güvenlik Sınırı (> 2.0 m)
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(time_steps, min_distances_m, color="#27ae60", linewidth=2.0, label="Minimum İHA-İHA Mesafesi (m)")
        ax2.axhline(2.0, color="#e74c3c", linestyle="--", label="Çarpışma Sınırı (2.0 m)")
        ax2.set_title("2. Sürü İçi Çarpışma Kaçınma Mesafesi (m)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("Zaman Adımı", fontsize=8)
        ax2.set_ylabel("Mesafe (Metre)", fontsize=8)
        ax2.legend(loc="upper right", fontsize=7)
        ax2.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 3: Hız Hizalanma Varyansı (Velocity Alignment Consensus)
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(time_steps, velocity_variances, color="#8e44ad", linewidth=2.0, label="Hız Varyansı σ² (m²/s²)")
        ax3.set_title("3. Hız Hizalanma Mutabakatı (Consensus)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Zaman Adımı", fontsize=8)
        ax3.set_ylabel("Varyans (m²/s²)", fontsize=8)
        ax3.legend(loc="upper right", fontsize=7)
        ax3.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 4: Sürü Ağırlık Merkezi (Center of Mass) İlerlemesi
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        com_traj = np.mean(drone_trajectories, axis=1) # (T, 3)
        dist_to_goal = [np.linalg.norm(com_traj[t] - target_waypoint) for t in range(len(com_traj))]
        ax4.plot(time_steps, dist_to_goal, color="#3498db", linewidth=2.0, label="Hedefe Kalan Mesafe (m)")
        ax4.set_title("4. Sürü Ağırlık Merkezi Hedef Yakınsaması", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_xlabel("Zaman Adımı", fontsize=8)
        ax4.set_ylabel("Hedefe Mesafe (m)", fontsize=8)
        ax4.legend(loc="upper right", fontsize=7)
        ax4.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 5: İHA Sürüsü 2D Dağılım Kesiti (XY Düzlemi)
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        for i in range(num_drones):
            ax5.plot(drone_trajectories[:, i, 0], drone_trajectories[:, i, 1], alpha=0.4)
            ax5.plot(drone_trajectories[-1, i, 0], drone_trajectories[-1, i, 1], "bo", markersize=4)
        ax5.plot(target_waypoint[0], target_waypoint[1], "r*", markersize=12, label="Hedef")
        ax5.set_title("5. Sürü Yatay Düzlem (XY) Dağılımı", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_xlabel("X (m)", fontsize=8)
        ax5.set_ylabel("Y (m)", fontsize=8)
        ax5.legend(loc="lower right", fontsize=7)
        ax5.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 6: Merkeziyetsiz GNN Sürü Hazır Bulunurluk Skoru
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        metrics_list = ["Sürü Güvenliği", "Hız Mutabakatı", "Hedefe Varış", "GNN Otonomisi"]
        scores = [
            profiler_metrics.get("safety_score", 100.0),
            profiler_metrics.get("alignment_score", 97.5),
            profiler_metrics.get("goal_reach_score", 98.0),
            profiler_metrics.get("swarm_flocking_readiness", 98.5)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#27ae60", alpha=0.85)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. GNN İHA Sürü Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
