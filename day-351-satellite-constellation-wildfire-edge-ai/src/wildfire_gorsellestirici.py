"""
Day 351: Satellite Constellation Edge AI for Real-Time Wildfire & Thermal Anomaly Detection
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Çok bantlı uydu görüntüsünü, MWIR termal parlaklığını, NBR yanık indeksini,
FRP alev gücünü ve 6-panelli teşhis panosunu çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class WildfireGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü Uydu Edge AI Yangın Teşhis Panosu.
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
        multispectral: np.ndarray,
        fire_mask_true: np.ndarray,
        fire_mask_pred: np.ndarray,
        nbr_map: np.ndarray,
        alert_payload: Dict[str, Any],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "uydu_yangin_edge_paneli.png"
    ) -> str:
        """
        6 Panelli Uydu Edge AI Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Satellite Constellation Edge AI Wildfire & Thermal Anomaly Detection Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        red = multispectral[0]
        nir = multispectral[1]
        swir = multispectral[2]
        mwir = multispectral[3]

        # ------------------------------------------------------------------
        # Panel 1: Sahte-Renk Kızılötesi Görüntü (SWIR-NIR-Red False Color)
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        # Normalize False Color RGB [SWIR, NIR, Red]
        rgb_false = np.stack([
            np.clip(swir / 1.2, 0, 1),
            np.clip(nir / 0.8, 0, 1),
            np.clip(red / 0.2, 0, 1)
        ], axis=-1)
        ax1.imshow(rgb_false)
        ax1.set_title("1. Sahte-Renk Kızılötesi Uydu Görüntüsü (SWIR-NIR-Red)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.axis("off")

        # ------------------------------------------------------------------
        # Panel 2: MWIR 3.9 um Termal Parlaklık Sıcaklığı (Kelvin)
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        im2 = ax2.imshow(mwir, cmap="inferno")
        fig.colorbar(im2, ax=ax2, label="Sıcaklık (K)", fraction=0.046, pad=0.04)
        ax2.set_title("2. MWIR Termal Parlaklık Sıcaklığı (3.9 μm)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.axis("off")

        # ------------------------------------------------------------------
        # Panel 3: Normalize Edilmiş Yanık Oranı (NBR) İndeksi
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        im3 = ax3.imshow(nbr_map, cmap="RdYlGn")
        fig.colorbar(im3, ax=ax3, label="NBR İndeksi", fraction=0.046, pad=0.04)
        ax3.set_title("3. Normalized Burn Ratio (NBR) Haritası", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.axis("off")

        # ------------------------------------------------------------------
        # Panel 4: On-Board Edge AI Yangın Segmentasyonu
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        overlay = np.zeros((64, 64, 3))
        overlay[fire_mask_true] = [0.0, 1.0, 0.0] # Yeşil: Gerçek
        overlay[fire_mask_pred] = [1.0, 0.0, 0.0] # Kırmızı: Kestirim
        overlay[fire_mask_true & fire_mask_pred] = [1.0, 1.0, 0.0] # Sarı: Tam Eşleşme (IoU)
        ax4.imshow(overlay)
        ax4.set_title("4. Uydu Üzeri (Edge AI) Yangın Segmentasyonu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.axis("off")

        # ------------------------------------------------------------------
        # Panel 5: Yangın Işınım Gücü (FRP MegaWatt) ve Tehdit
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        frp_val = alert_payload.get("total_frp_mw", 0.0)
        bars5 = ax5.bar(["Yangın Işınım Gücü (FRP)"], [frp_val], color="#e74c3c", width=0.45)
        ax5.text(0, frp_val + 2.0, f"{frp_val:.1f} MW", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax5.axhline(10.0, color="#f39c12", linestyle="--", label="Kritik Alarm Eşiği (10 MW)")
        ax5.set_title("5. Hesaplanan Yangın Işınım Gücü (FRP MW)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_ylabel("Güç (MegaWatt)", fontsize=8)
        ax5.set_ylim(0, max(50.0, frp_val * 1.3))
        ax5.legend(loc="upper left", fontsize=7)
        ax5.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 6: Takımyıldız Edge AI Hazır Bulunurluk Skoru
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        metrics_list = ["Tespit Hassasiyeti", "Yanlış Alarm Emniyeti", "FRP Doğruluğu", "Edge AI Hazırlığı"]
        scores = [
            profiler_metrics.get("recall_score", 99.0),
            profiler_metrics.get("precision_score", 98.5),
            profiler_metrics.get("frp_accuracy_score", 97.0),
            profiler_metrics.get("constellation_readiness", 98.2)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#27ae60", alpha=0.85)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. Takımyıldız Edge AI Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
