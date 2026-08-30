"""
Day 328: SNN-ANN Hybrid Transduction Layers
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; ANN aktivasyonlarını, dönüştürülen zamansal spike akışlarını,
LIF zar potansiyel entegrasyonunu, SNN-to-ANN süzgeçli vektörlerini ve hibrit ağ teşhis panosunu barındırır.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class HybridGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü SNN-ANN Hibrit Transdüksiyon Teşhis ve Performans Panosu.
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
        ann_feature_map: np.ndarray,
        spike_stream: np.ndarray,
        v_mem_history: np.ndarray,
        snn_to_ann_features: np.ndarray,
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "hibrit_transduksiyon_paneli.png"
    ) -> str:
        """
        6 Panelli SNN-ANN Hibrit Ağ Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.suptitle(
            "SNN-ANN Hybrid Transduction Layers & Edge Energy Profilleme Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        # ------------------------------------------------------------------
        # Panel 1: Sürekli ANN Girdi Aktivasyon Haritası
        # ------------------------------------------------------------------
        ax1 = axes[0, 0]
        im1 = ax1.imshow(ann_feature_map[:8, :16], cmap="viridis", aspect="auto")
        plt.colorbar(im1, ax=ax1, label="ANN ReLU Aktivasyon Değeri")
        ax1.set_title("1. Sürekli ANN Girdi Aktivasyonu (Continuous Float)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("Öznitelik İndeksi", fontsize=8)
        ax1.set_ylabel("Örnek (Batch) İndeksi", fontsize=8)

        # ------------------------------------------------------------------
        # Panel 2: ANN-to-SNN Dönüştürülen Spike Akışı Raster Diyagramı
        # ------------------------------------------------------------------
        ax2 = axes[0, 1]
        t_steps, n_snn = spike_stream.shape[1], spike_stream.shape[2]
        sample_spikes = spike_stream[0]  # (Time, Neurons)
        
        for neuron_i in range(min(16, n_snn)):
            spike_times = np.where(sample_spikes[:, neuron_i] > 0)[0]
            ax2.vlines(spike_times, neuron_i + 0.6, neuron_i + 1.4, colors="#27ae60", linewidth=1.5)
        ax2.set_title("2. Transduced ANN-to-SNN Spike Raster (Poisson Stream)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("Zaman Adımı (t)", fontsize=8)
        ax2.set_ylabel("SNN Nöron İndeksi", fontsize=8)
        ax2.set_ylim(0, min(17, n_snn + 1))
        ax2.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 3: SNN LIF Zar Potansiyeli Entegrasyonu V(t)
        # ------------------------------------------------------------------
        ax3 = axes[0, 2]
        t_vec = np.arange(t_steps)
        sample_vmem = v_mem_history[0]  # (Time, Neurons)
        for n in range(min(4, n_snn)):
            ax3.plot(t_vec, sample_vmem[:, n], label=f"Nöron {n+1} V(t)", linewidth=1.8)
        ax3.axhline(1.0, color="#e74c3c", linestyle="--", label="Eşik (V_th=1.0)")
        ax3.set_title("3. SNN LIF Zar Potansiyeli Entegrasyonu V(t)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Zaman Adımı (t)", fontsize=8)
        ax3.set_ylabel("Zar Potansiyeli V", fontsize=8)
        ax3.legend(loc="upper right", fontsize=7)
        ax3.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 4: SNN-to-ANN Düşük Geçiren Süzgeçli Aktivasyon Vektörü
        # ------------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.bar(np.arange(min(16, snn_to_ann_features.shape[1])), snn_to_ann_features[0, :16], color="#8e44ad", alpha=0.8, width=0.5)
        ax4.set_title("4. Transduced SNN-to-ANN Aktivasyon Vektörü (Low-Pass Filter)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_xlabel("ANN Öznitelik İndeksi", fontsize=8)
        ax4.set_ylabel("Aktivasyon Genliği", fontsize=8)
        ax4.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 5: Mimari Sınıflandırma Başarımı (ANN vs SNN vs Hibrit)
        # ------------------------------------------------------------------
        ax5 = axes[1, 1]
        architectures = ["Saf ANN (Dense)", "Saf SNN (LIF)", "SNN-ANN Hibrit (Bizim)"]
        accuracies = [
            profiler_metrics.get("pure_ann_acc", 94.0),
            profiler_metrics.get("pure_snn_acc", 88.5),
            profiler_metrics.get("hybrid_acc", 96.8)
        ]
        bars = ax5.bar(architectures, accuracies, color=["#2980b9", "#e67e22", "#27ae60"], width=0.5, alpha=0.85)
        for bar in bars:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"%{yval:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax5.set_title("5. Sınıflandırma Başarımı (ANN vs SNN vs Hibrit)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_ylabel("Test Doğruluğu (%)", fontsize=8)
        ax5.set_ylim(0, 115)
        ax5.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 6: Edge Donanım Enerji Tasarrufu ve Dönüştürme Verimliliği
        # ------------------------------------------------------------------
        ax6 = axes[1, 2]
        metrics_list = ["Dönüştürme Sadakati", "Spike Seyreklilik Kazancı", "Edge Enerji Verimliliği", "Hibrit Sistem Skoru"]
        scores = [
            profiler_metrics.get("transduction_fidelity_score", 96.0),
            profiler_metrics.get("spike_sparsity_score", 85.0),
            profiler_metrics.get("edge_energy_score", 92.0),
            profiler_metrics.get("hybrid_system_score", 95.5)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#3498db", alpha=0.8)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. Edge Donanım Transdüksiyon Performansı", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
