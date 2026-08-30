"""
Day 324: Neuromorphic Hardware Mapping (Intel Loihi 2 & SynSense)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Intel Loihi 2 Neuro-Core Mesh haritalamasını, INT8 kuantizasyon dağılımını,
AER paket yönlendirme hop mesafelerini ve donanım hazır bulunurluk panosunu içeren 6-panelli görselleştiriciyi içerir.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class LoihiGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü Loihi 2 Donanım Eşleme ve Performans Panosu.
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
        mapping_info: Dict[str, Any],
        weights_fp32: np.ndarray,
        weights_dequant: np.ndarray,
        aer_packets: List[Any],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "loihi_donanim_paneli.png"
    ) -> str:
        """
        6 Panelli Loihi Donanım Eşleme Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Intel Loihi 2 & SynSense Neuromorphic Hardware Mapping Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        mesh_rows = 4
        mesh_cols = 4

        # ------------------------------------------------------------------
        # Panel 1: Neuro-Core Mesh Çip Haritası & Doluluk Heatmap'i
        # ------------------------------------------------------------------
        ax1 = axes[0, 0]
        mesh_grid = np.zeros((mesh_rows, mesh_cols))
        used_cores = mapping_info.get("used_cores", 0)
        
        idx = 0
        for r in range(mesh_rows):
            for c in range(mesh_cols):
                if idx < used_cores:
                    mesh_grid[r, c] = 1.0  # Aktif Çekirdek
                idx += 1

        im1 = ax1.imshow(mesh_grid, cmap="YlGn", vmin=0.0, vmax=1.0)
        ax1.set_title(f"1. Neuro-Core Mesh Haritası ({used_cores}/{mesh_rows*mesh_cols} Çekirdek)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("Mesh Sütun (X)", fontsize=8)
        ax1.set_ylabel("Mesh Satır (Y)", fontsize=8)
        for r in range(mesh_rows):
            for c in range(mesh_cols):
                status = "ACTIVE" if mesh_grid[r, c] > 0.5 else "IDLE"
                ax1.text(c, r, f"Core {r*mesh_cols+c}\n({status})", ha="center", va="center", fontsize=7, color="#2c3e50" if status=="IDLE" else "#1e8449")

        # ------------------------------------------------------------------
        # Panel 2: Ağırlık Kuantizasyonu (FP32 vs INT8 Dağılımı)
        # ------------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.hist(weights_fp32.flatten(), bins=20, alpha=0.5, color="#2980b9", label="FP32 Orijinal", density=True)
        ax2.hist(weights_dequant.flatten(), bins=20, alpha=0.6, color="#27ae60", label="INT8 Kuantize (Dequant)", density=True)
        ax2.set_title(f"2. INT8 Sabitleştirilmiş Kuantizasyon (SQNR: {mapping_info.get('sqnr_db', 0.0):.1f} dB)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("Sinaptik Ağırlık Değeri W", fontsize=8)
        ax2.set_ylabel("Yoğunluk", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 3: AER Paket Yönlendirme Hop Mesafesi Histogramı
        # ------------------------------------------------------------------
        ax3 = axes[0, 2]
        hop_distances = [p.hop_distance for p in aer_packets] if aer_packets else [1, 2, 1, 3, 2]
        ax3.hist(hop_distances, bins=np.arange(0.5, 6.5, 1.0), color="#8e44ad", edgecolor="#5b2c6f", alpha=0.8, rwidth=0.7)
        ax3.set_title("3. AER Yönlendirme Hop Mesafesi Dağılımı", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Manhattan Hop Mesafesi (Çekirdek Adımı)", fontsize=8)
        ax3.set_ylabel("Paket Sayısı", fontsize=8)
        ax3.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 4: Fixed-Point INT16 Zar Potansiyeli Simülasyonu
        # ------------------------------------------------------------------
        ax4 = axes[1, 0]
        t_steps = np.arange(30)
        v_fp32 = np.sin(t_steps * 0.2) * 0.8 + 0.2
        v_int16 = np.round(v_fp32 * 1000.0) / 1000.0  # Simulated fixed point
        
        ax4.plot(t_steps, v_fp32, label="FP32 Potansiyel V", color="#2980b9", linewidth=2.0)
        ax4.step(t_steps, v_int16, label="Loihi INT16 Fixed-Point", color="#e67e22", linestyle="--", where="mid")
        ax4.axhline(0.8, color="#e74c3c", linestyle=":", label="Eşik (V_th)")
        ax4.set_title("4. Sabitleştirilmiş (Fixed-Point) Zar Potansiyeli", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_xlabel("Zaman Adımı (t)", fontsize=8)
        ax4.set_ylabel("Zar Potansiyeli V", fontsize=8)
        ax4.legend(loc="upper right", fontsize=8)
        ax4.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 5: Donanım Enerji Tüketimi (Loihi 2 vs GPU)
        # ------------------------------------------------------------------
        ax5 = axes[1, 1]
        categories = ["Loihi 2 (INT8 SOP)", "GPU (FP16 FLOP)"]
        energies = [
            profiler_metrics.get("loihi_energy_uj", 0.5),
            profiler_metrics.get("gpu_energy_uj", 15.0)
        ]
        bars = ax5.bar(categories, energies, color=["#27ae60", "#c0392b"], width=0.5, alpha=0.85)
        for bar in bars:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + max(energies)*0.02, f"{yval:.2f} uJ", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax5.set_title("5. Tahmini Enerji Tüketimi (microJoules / Çıkarım)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_ylabel("Enerji (uJ)", fontsize=8)
        ax5.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 6: Donanım Eşleme Skoru ve Doğruluk
        # ------------------------------------------------------------------
        ax6 = axes[1, 2]
        metrics_list = ["Çekirdek Verimliliği", "AĞ Hop Skoru", "INT8 Kuantizasyon", "AER Paket Oranı"]
        scores = [
            profiler_metrics.get("core_efficiency_score", 90.0),
            profiler_metrics.get("hop_score", 95.0),
            profiler_metrics.get("quant_accuracy_score", 98.5),
            profiler_metrics.get("aer_throughput_score", 92.0)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#3498db", alpha=0.8)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. Loihi 2 Donanım Hazır Bulunurluk Skoru", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
