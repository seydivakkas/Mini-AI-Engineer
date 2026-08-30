"""
Day 356: Autonomous Aerial Refueling (AAR) Vision-Based Docking Flight Controller
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; 3D havada yakıt ikmali yaklaşma koridorunu, sepet düzlemi hedef dairesini,
yanal hata düşüşünü ve 6-panelli teşhis panosunu çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class AARGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü Otonom Havada Yakıt İkmali (AAR) Teşhis Panosu.
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
        mission_res: Dict[str, Any],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "aar_yakit_ikmal_paneli.png"
    ) -> str:
        """
        6 Panelli AAR Kenetlenme Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Autonomous Aerial Refueling (AAR) Vision-Based Docking Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        time_steps = mission_res["time_steps"]
        uav_traj = mission_res["uav_trajectory"]
        drogue_traj = mission_res["drogue_trajectory"]
        miss_cm = mission_res["miss_distances_cm"]
        docking_time = mission_res["docking_time_sec"]

        # ------------------------------------------------------------------
        # Panel 1: 3D Havada Yakıt İkmali Yaklaşma Koridoru
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(drogue_traj[:, 0], drogue_traj[:, 1], drogue_traj[:, 2], "r-", linewidth=2.0, label="Tanker Sepeti (Drogue)")
        ax1.plot(uav_traj[:, 0], uav_traj[:, 1], uav_traj[:, 2], "b--", linewidth=1.8, label="İHA Probu (Receiver)")
        ax1.scatter(drogue_traj[-1, 0], drogue_traj[-1, 1], drogue_traj[-1, 2], color="#27ae60", s=100, marker="o", label="Kenetlenme Noktası")
        ax1.set_title("1. 3D Havada Yakıt İkmali Yaklaşma Yolu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("X (İleri m)", fontsize=7)
        ax1.set_ylabel("Y (Yanal m)", fontsize=7)
        ax1.set_zlabel("Z (İrtifa m)", fontsize=7)
        ax1.legend(loc="upper left", fontsize=6)

        # ------------------------------------------------------------------
        # Panel 2: Sepet Düzleminde (YZ) Hedef Dairesi & Hizalama
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        theta = np.linspace(0, 2*np.pi, 100)
        # 10 cm Güvenli Sepet Ağzı Dairesi
        ax2.plot(10.0 * np.cos(theta), 10.0 * np.sin(theta), "g--", linewidth=2.0, label="Güvenli Yakalama Zarfı (10 cm)")
        ax2.plot(35.0 * np.cos(theta), 35.0 * np.sin(theta), "r:", linewidth=1.5, label="Acil Ayrılma Limiti (35 cm)")
        
        rel_y_cm = (uav_traj[:, 1] - drogue_traj[:, 1]) * 100.0
        rel_z_cm = (uav_traj[:, 2] - drogue_traj[:, 2]) * 100.0
        ax2.plot(rel_y_cm, rel_z_cm, "b-", alpha=0.6, linewidth=1.2, label="İHA Probunun Hizalanması")
        ax2.scatter(rel_y_cm[-1], rel_z_cm[-1], color="#27ae60", s=100, marker="*", label="Temas Noktası")
        ax2.set_title("2. Sepet Düzleminde (YZ) Prob Hizalama Hassasiyeti", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("Yanal Hata Y (cm)", fontsize=8)
        ax2.set_ylabel("Dikey Hata Z (cm)", fontsize=8)
        ax2.legend(loc="lower right", fontsize=6)
        ax2.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 3: Boylamasına Yaklaşma Mesafesi (X Ekseni)
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        rel_x = drogue_traj[:, 0] - uav_traj[:, 0]
        ax3.plot(time_steps, rel_x, color="#2980b9", linewidth=2.0, label="Sepete Göreli X Mesafesi (m)")
        if docking_time != -1:
            ax3.axvline(docking_time, color="#27ae60", linestyle="--", label=f"Kenetlenme ({docking_time:.1f}s)")
        ax3.set_title("3. Boylamasına Yaklaşma Mesafesi (m)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Zaman (saniye)", fontsize=8)
        ax3.set_ylabel("Mesafe (Metre)", fontsize=8)
        ax3.legend(loc="upper right", fontsize=7)
        ax3.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 4: Yanal Iskalama Mesafesi Düşüşü (cm)
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(time_steps, miss_cm, color="#8e44ad", linewidth=2.0, label="Toplam Yanal Sapma (cm)")
        ax4.axhline(8.0, color="#27ae60", linestyle="--", label="Hedef Tolerans (< 8 cm)")
        ax4.set_title("4. Yanal Iskalama Mesafesi ve Kapanış (cm)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_xlabel("Zaman (saniye)", fontsize=8)
        ax4.set_ylabel("Yanal Hata (cm)", fontsize=8)
        ax4.legend(loc="upper right", fontsize=7)
        ax4.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 5: Son Temas Hassasiyet Karşılaştırması
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        final_err = mission_res["final_lateral_error_cm"]
        categories = ["Elde Edilen Temas Hassasiyeti", "Askeri Standart Limiti"]
        vals = [final_err, 8.0]
        bars5 = ax5.bar(categories, vals, color=["#27ae60", "#c0392b"], width=0.45)
        for bar in bars5:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f} cm", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax5.set_title("5. Son Kenetlenme Temas Hassasiyeti (cm)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_ylabel("Hata (cm)", fontsize=8)
        ax5.set_ylim(0, 12.0)
        ax5.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 6: Otonom AAR Görev Hazır Bulunurluk Skoru
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        metrics_list = ["Görü Tabanlı Takip", "Girdap Bastırma", "Hizalama Hassasiyeti", "AAR Görev Başarısı"]
        scores = [
            profiler_metrics.get("vision_tracking_score", 100.0),
            profiler_metrics.get("vortex_rejection_score", 98.5),
            profiler_metrics.get("docking_precision_score", 99.0),
            profiler_metrics.get("aar_mission_success_score", 99.2)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#27ae60", alpha=0.85)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. Otonom Havada İkmal (AAR) Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
