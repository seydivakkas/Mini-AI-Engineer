"""
Day 326: Intracortical Spike Sorting & LFADS Latent Dynamics
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; MEA ham gerilim sinyallerini, PCA+GMM spike ayrıştırma seyreklik kümesini,
spike raster grafiğini, LFADS latent yörüngelerini ve rekonstrüksiyon panosunu içerir.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class LFADSGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü LFADS & Spike Sorting Teşhis ve Performans Panosu.
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
        raw_voltage: np.ndarray,
        spike_indices: np.ndarray,
        waveforms_2d: np.ndarray,
        cluster_labels: np.ndarray,
        waveforms: np.ndarray,
        spikes_raster: np.ndarray,
        latent_factors: np.ndarray,
        reconstructed_rates: np.ndarray,
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "lfads_spike_paneli.png"
    ) -> str:
        """
        6 Panelli LFADS Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Intracortical Spike Sorting & LFADS Latent Population Dynamics Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        t_raw = np.arange(len(raw_voltage)) / 30000.0 * 1000.0  # ms

        # ------------------------------------------------------------------
        # Panel 1: Ham Sinyal ve Negatif Pik Eşik Tespiti
        # ------------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.plot(t_raw[:1500], raw_voltage[:1500], color="#2c3e50", alpha=0.7, label="Filtrelenmiş Sinyal")
        th_val = -4.0 * np.std(raw_voltage)
        ax1.axhline(th_val, color="#e74c3c", linestyle="--", label=f"Spike Eşiği ({th_val:.1f} uV)")
        
        valid_spikes = [idx for idx in spike_indices if idx < 1500]
        if valid_spikes:
            ax1.scatter(t_raw[valid_spikes], raw_voltage[valid_spikes], color="#c0392b", s=40, zorder=5, label="Tespit Edilen Pik")
        ax1.set_title("1. Ham MEA Gerilimi ve Spike Eşik Tespiti", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("Zaman (ms)", fontsize=8)
        ax1.set_ylabel("Genlik (uV)", fontsize=8)
        ax1.legend(loc="upper right", fontsize=7)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 2: PCA 2D Spike Kümeleri (Single-Unit Sorting)
        # ------------------------------------------------------------------
        ax2 = axes[0, 1]
        colors = ["#e74c3c", "#2980b9", "#27ae60", "#f39c12"]
        n_clusters = len(np.unique(cluster_labels))
        for c in range(n_clusters):
            mask = (cluster_labels == c)
            ax2.scatter(waveforms_2d[mask, 0], waveforms_2d[mask, 1], color=colors[c % len(colors)], label=f"Nöron {c+1}", s=25, alpha=0.8)
        ax2.set_title("2. Spike Waveform PCA 2D Kümeleme (GMM)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("PCA Bileşen 1", fontsize=8)
        ax2.set_ylabel("PCA Bileşen 2", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 3: Nöron Birimlerinin Ortalama Aksiyon Potansiyel Dalgası
        # ------------------------------------------------------------------
        ax3 = axes[0, 2]
        t_wave = np.arange(waveforms.shape[1]) / 30.0  # ms
        for c in range(n_clusters):
            mask = (cluster_labels == c)
            if np.sum(mask) > 0:
                mean_w = np.mean(waveforms[mask], axis=0)
                std_w = np.std(waveforms[mask], axis=0)
                ax3.plot(t_wave, mean_w, color=colors[c % len(colors)], linewidth=2.0, label=f"Unit {c+1} Waveform")
                ax3.fill_between(t_wave, mean_w - std_w, mean_w + std_w, color=colors[c % len(colors)], alpha=0.15)
        ax3.set_title("3. Nöron Birimlerinin Dalgabiçim (Waveform) Profili", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Pencere Zamanı (ms)", fontsize=8)
        ax3.set_ylabel("Volt (uV)", fontsize=8)
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 4: Nöron Popülasyonu Spike Raster Grafiği
        # ------------------------------------------------------------------
        ax4 = axes[1, 0]
        n_neurons, t_len = spikes_raster.shape
        for i in range(min(15, n_neurons)):
            spike_times = np.where(spikes_raster[i] > 0)[0]
            ax4.vlines(spike_times, i + 0.6, i + 1.4, colors="#2c3e50", linewidth=1.2)
        ax4.set_title("4. Nöron Popülasyonu Spike Raster Diyagramı", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_xlabel("Zaman Adımı (t)", fontsize=8)
        ax4.set_ylabel("Nöron İndeksi", fontsize=8)
        ax4.set_ylim(0, min(16, n_neurons + 1))
        ax4.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 5: LFADS Latent Yörüngeleri g(t)
        # ------------------------------------------------------------------
        ax5 = axes[1, 1]
        t_steps = np.arange(latent_factors.shape[0])
        for f in range(min(4, latent_factors.shape[1])):
            ax5.plot(t_steps, latent_factors[:, f], label=f"Latent Faktör {f+1}", linewidth=1.8)
        ax5.set_title("5. LFADS Latent Dinamikleri g(t) Yörüngesi", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_xlabel("Zaman Adımı (t)", fontsize=8)
        ax5.set_ylabel("Latent Faktör Genliği", fontsize=8)
        ax5.legend(loc="upper right", fontsize=7)
        ax5.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 6: LFADS Düzleştirilmiş Ateşleme Oranı lambda(t)
        # ------------------------------------------------------------------
        ax6 = axes[1, 2]
        sample_neuron_rate = reconstructed_rates[:, 0]
        sample_neuron_spikes = spikes_raster[0]
        ax6.bar(t_steps, sample_neuron_spikes, color="#bdc3c7", alpha=0.6, label="Ham Spike Sayımı")
        ax6.plot(t_steps, sample_neuron_rate, color="#9b59b6", linewidth=2.2, label="LFADS Tahmini Rate lambda(t)")
        ax6.set_title(f"6. Pürüzsüz Ateşleme Oranı (Poisson Loss: {profiler_metrics.get('poisson_loss', 0.0):.3f})", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Zaman Adımı (t)", fontsize=8)
        ax6.set_ylabel("Ateşleme Oranı (Hz / Bin)", fontsize=8)
        ax6.legend(loc="upper right", fontsize=8)
        ax6.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
