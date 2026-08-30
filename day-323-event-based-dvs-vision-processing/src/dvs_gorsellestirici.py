"""
Day 323: Dynamic Vision Sensors (DVS) & Event-Based Processing
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; 3D DVS olay akışı saçılım grafiğini, SAE zamansal sönümlenme yüzeyini,
Voxel Grid dilimlerini ve veri sıkıştırma kazancı metriklerini içeren 6-panelli teşhis panosunu barındırır.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import torch


class DVSGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü DVS Olay Tabanlı Görsel İşleme Teşhis Panosu.
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
        events: np.ndarray,
        sae_surface: np.ndarray,
        voxel_grid: torch.Tensor,
        train_losses: List[float],
        test_accs: List[float],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "dvs_isleme_paneli.png"
    ) -> str:
        """
        6 Panelli DVS Görsel İşleme Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Dynamic Vision Sensors (DVS) & Olay Tabanlı Görsel İşleme Teşhis Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        # ------------------------------------------------------------------
        # Panel 1: 3D Olay Akışı Saçılım Grafiği (x, y, t_us)
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1, projection="3d")
        if len(events) > 0:
            x, y, t, p = events[:, 0], events[:, 1], events[:, 2] / 1000.0, events[:, 3]
            colors = np.where(p > 0, "#e74c3c", "#3498db")
            ax1.scatter(x, y, t, c=colors, s=6, alpha=0.6)
        ax1.set_title("1. 3D DVS Olay Akışı (x, y, t_ms)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("X (px)", fontsize=8)
        ax1.set_ylabel("Y (px)", fontsize=8)
        ax1.set_zlabel("Zaman (ms)", fontsize=8)

        # ------------------------------------------------------------------
        # Panel 2: Polarite Bazlı Uzamsal Olay Haritası (+1 Red / -1 Blue)
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        pos_events = events[events[:, 3] > 0]
        neg_events = events[events[:, 3] < 0]
        
        if len(pos_events) > 0:
            ax2.scatter(pos_events[:, 0], pos_events[:, 1], color="#e74c3c", s=10, alpha=0.7, label="ON Event (+1)")
        if len(neg_events) > 0:
            ax2.scatter(neg_events[:, 0], neg_events[:, 1], color="#3498db", s=10, alpha=0.7, label="OFF Event (-1)")
        
        ax2.set_title("2. Polarite Haritası (ON / OFF Events)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("X Piksel", fontsize=8)
        ax2.set_ylabel("Y Piksel", fontsize=8)
        ax2.set_aspect("equal")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 3: Surface of Active Events (SAE) Sönümlenme Yüzeyi
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        sae_combined = sae_surface[0] + sae_surface[1]  # Combine polarities
        im3 = ax3.imshow(sae_combined, cmap="magma", aspect="auto")
        plt.colorbar(im3, ax=ax3, label="Üstel Sönümlenme S(x, y)")
        ax3.set_title("3. Surface of Active Events (SAE) Yüzeyi", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("X Piksel", fontsize=8)
        ax3.set_ylabel("Y Piksel", fontsize=8)

        # ------------------------------------------------------------------
        # Panel 4: Voxel Grid Zamansal Dilimleri
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        voxel_np = voxel_grid.cpu().numpy()  # (C, H, W)
        voxel_slice = np.sum(voxel_np[:voxel_np.shape[0]//2], axis=0)  # Positive bins sum
        im4 = ax4.imshow(voxel_slice, cmap="viridis", aspect="auto")
        plt.colorbar(im4, ax=ax4, label="Olay Sayısı Bin")
        ax4.set_title("4. 3D Voxel Grid Dilim Yoğunluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_xlabel("X Piksel", fontsize=8)
        ax4.set_ylabel("Y Piksel", fontsize=8)

        # ------------------------------------------------------------------
        # Panel 5: Veri Hacmi & Sıkıştırma Kazancı (DVS vs Kare Kamera)
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        categories = ["Standart Video (Frame)", "DVS Olay Akışı"]
        bytes_values = [
            profiler_metrics.get("frame_bytes", 153600),
            profiler_metrics.get("dvs_bytes", 9600)
        ]
        bars = ax5.bar(categories, [b / 1024.0 for b in bytes_values], color=["#e67e22", "#27ae60"], width=0.5, alpha=0.85)
        for bar in bars:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f} KB", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax5.set_title("5. Veri Hacmi Karşılaştırması (KB)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_ylabel("Veri Boyutu (KB)", fontsize=8)
        ax5.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 6: Spiking Event ConvNet Eğitim & Doğruluk Grafiği
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        epochs_arr = np.arange(1, len(train_losses) + 1)
        ax6_twin = ax6.twinx()
        
        l1 = ax6.plot(epochs_arr, train_losses, color="#2980b9", linewidth=2.0, label="Kayıp (Loss)")
        l2 = ax6_twin.plot(epochs_arr, test_accs, color="#27ae60", linewidth=2.0, linestyle="--", label="Test Doğruluğu (%)")
        
        ax6.set_title("6. Spiking Event ConvNet Başarımı", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Epok", fontsize=8)
        ax6.set_ylabel("Cross-Entropy Kaybı", fontsize=8, color="#2980b9")
        ax6_twin.set_ylabel("Doğruluk (%)", fontsize=8, color="#27ae60")
        
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc="center right", fontsize=8)
        ax6.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
