"""
Day 321: Spiking Neural Networks (SNN) & Leaky Integrate-and-Fire (LIF) Neuron Mathematics
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; SNN modelinin nöron seviyesinde zar potansiyellerini, spike raster grafiklerini,
seyreklik (sparsity) metriklerini ve surrogate gradient profillerini kapsayan
6-panelli teşhis görselleştirme aracını içerir.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch


class SNNGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü SNN Teşhis ve Performans Panosu Görsellestirici.
    """
    def __init__(self, cikti_dizini: str = None):
        if cikti_dizini is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cikti_dizini = os.path.join(base_dir, "ciktilar")
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

        # Stil Ayarları
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI", "Arial"]
        plt.rcParams["axes.edgecolor"] = "#2c3e50"
        plt.rcParams["axes.linewidth"] = 1.2

    def teshis_panelini_ciz(
        self,
        info_dict: Dict[str, Any],
        train_losses: List[float],
        test_accuracies: List[float],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "snn_lif_teshis_paneli.png"
    ) -> str:
        """
        6 Panelli Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Spiking Neural Networks (SNN) & LIF Nöron Teşhis Panosu",
            fontsize=16,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        mem1 = info_dict["mem1"][0].detach().cpu().numpy()  # (T, Hidden)
        spikes1 = info_dict["spikes1"][0].detach().cpu().numpy()  # (T, Hidden)
        time_steps = mem1.shape[0]

        # ------------------------------------------------------------------
        # Panel 1: Zar Potansiyeli Zaman Serisi (Membrane Potential V(t))
        # ------------------------------------------------------------------
        ax1 = axes[0, 0]
        t_axis = np.arange(time_steps)
        # İlk 3 nöronun potansiyel değişimini çiz
        for i in range(min(3, mem1.shape[1])):
            ax1.plot(t_axis, mem1[:, i], label=f"Nöron {i+1}", linewidth=1.8, alpha=0.85)
        ax1.axhline(y=1.0, color="#e74c3c", linestyle="--", linewidth=1.5, label="Eşik (V_th=1.0)")
        ax1.set_title("1. Zar Potansiyeli Zaman Serisi V(t)", fontsize=11, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("Zaman Adımı (t)", fontsize=9)
        ax1.set_ylabel("Zar Potansiyeli (V)", fontsize=9)
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # ------------------------------------------------------------------
        # Panel 2: Spike Raster Plot (Gizli Katman Spike Zamanlamaları)
        # ------------------------------------------------------------------
        ax2 = axes[0, 1]
        t_indices, neuron_indices = np.where(spikes1 > 0.5)
        ax2.scatter(t_indices, neuron_indices, color="#2980b9", s=12, marker="|", alpha=0.9)
        ax2.set_title("2. Gizli Katman Spike Raster Plot", fontsize=11, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("Zaman Adımı (t)", fontsize=9)
        ax2.set_ylabel("Nöron İndeksi", fontsize=9)
        ax2.set_ylim(-0.5, mem1.shape[1] - 0.5)
        ax2.grid(True, linestyle=":", alpha=0.6)

        # ------------------------------------------------------------------
        # Panel 3: Ateşleme Oranı (Firing Rate) Dağılımı
        # ------------------------------------------------------------------
        ax3 = axes[0, 2]
        firing_rates = np.sum(spikes1, axis=0) / time_steps
        n_counts, n_bins, patches = ax3.hist(
            firing_rates, bins=12, color="#27ae60", edgecolor="#1e8449", alpha=0.75
        )
        ax3.set_title("3. Nöron Ateşleme Oranı (Firing Rate) Dağılımı", fontsize=11, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Ateşleme Oranı (Spikes / T)", fontsize=9)
        ax3.set_ylabel("Nöron Sayısı", fontsize=9)
        ax3.grid(True, linestyle=":", alpha=0.6)

        # ------------------------------------------------------------------
        # Panel 4: Enerji & Seyreklik (SNN vs ANN Karşılaştırması)
        # ------------------------------------------------------------------
        ax4 = axes[1, 0]
        categories = ["SNN (SOP)", "ANN (FLOP)"]
        energy_values = [
            profiler_metrics.get("snn_energy_pj", 10.0),
            profiler_metrics.get("ann_energy_pj", 100.0)
        ]
        bars = ax4.bar(categories, energy_values, color=["#8e44ad", "#e67e22"], width=0.5, alpha=0.85)
        for bar in bars:
            yval = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width()/2.0,
                yval + max(energy_values)*0.02,
                f"{yval:.1f} pJ",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )
        ax4.set_title("4. Tahmini Enerji Tüketimi (SNN vs ANN)", fontsize=11, fontweight="bold", color="#2c3e50")
        ax4.set_ylabel("Tahmini Enerji (picoJoules / Örnek)", fontsize=9)
        ax4.grid(True, linestyle=":", alpha=0.6, axis="y")

        # ------------------------------------------------------------------
        # Panel 5: Surrogate Gradient Türev Profili (Fast Sigmoid)
        # ------------------------------------------------------------------
        ax5 = axes[1, 1]
        v_diff = np.linspace(-1.5, 1.5, 300)
        slope = 25.0
        surrogate_grad = slope / ((1.0 + slope * np.abs(v_diff)) ** 2)
        ax5.plot(v_diff, surrogate_grad, color="#c0392b", linewidth=2.0, label="Fast Sigmoid dS/dV")
        ax5.axvline(x=0.0, color="#7f8c8d", linestyle="--", linewidth=1.2, label="V - V_th = 0")
        ax5.set_title("5. Surrogate Gradient Profili (dS/dV)", fontsize=11, fontweight="bold", color="#2c3e50")
        ax5.set_xlabel("Delta Potansiyel (V - V_th)", fontsize=9)
        ax5.set_ylabel("Gradyan Büyüklüğü", fontsize=9)
        ax5.legend(loc="upper right", fontsize=8)
        ax5.grid(True, linestyle=":", alpha=0.6)

        # ------------------------------------------------------------------
        # Panel 6: Sınıflandırma Eğitimi ve Test Doğruluğu
        # ------------------------------------------------------------------
        ax6 = axes[1, 2]
        epochs = np.arange(1, len(train_losses) + 1)
        ax6_twin = ax6.twinx()
        
        l1 = ax6.plot(epochs, train_losses, color="#2980b9", linewidth=2.0, label="Eğitim Kaybı (Loss)")
        l2 = ax6_twin.plot(epochs, test_accuracies, color="#27ae60", linewidth=2.0, linestyle="--", label="Test Doğruluğu (%)")
        
        ax6.set_title("6. SNN Eğitim & Test Performansı", fontsize=11, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Epok (Epoch)", fontsize=9)
        ax6.set_ylabel("Kayıp (Cross-Entropy)", fontsize=9, color="#2980b9")
        ax6_twin.set_ylabel("Doğruluk (%)", fontsize=9, color="#27ae60")
        
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc="center right", fontsize=8)
        ax6.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
