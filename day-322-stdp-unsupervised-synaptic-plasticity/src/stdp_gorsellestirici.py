"""
Day 322: Spike-Timing-Dependent Plasticity (STDP) & Unsupervised Learning
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; STDP denklem eğrilerini, sinaptik ağırlık dönüşüm matrislerini,
presinaptik/postsinaptik iz zaman serilerini ve WTA yarış panosunu içeren
6-panelli teşhis görselleştirme aracını barındırır.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import torch


class STDPGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü STDP Plastisite Teşhis ve Performans Panosu.
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
        initial_weights: np.ndarray,
        final_weights: np.ndarray,
        stdp_info: Dict[str, Any],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "stdp_plastisite_paneli.png"
    ) -> str:
        """
        6 Panelli STDP Plastisite Grafiğini Oluşturur ve Kaydeder.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Spike-Timing-Dependent Plasticity (STDP) & Denetimsiz Öğrenme Teşhis Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        # ------------------------------------------------------------------
        # Panel 1: Üstel STDP Öğrenme Eğrisi (Delta W vs Delta t)
        # ------------------------------------------------------------------
        ax1 = axes[0, 0]
        delta_t_pos = np.linspace(0.1, 50, 200)
        delta_t_neg = np.linspace(-50, -0.1, 200)
        
        a_plus, a_minus = 0.02, 0.015
        tau_plus, tau_minus = 20.0, 20.0
        
        dw_pos = a_plus * np.exp(-delta_t_pos / tau_plus)
        dw_neg = -a_minus * np.exp(delta_t_neg / tau_minus)
        
        ax1.plot(delta_t_pos, dw_pos, color="#27ae60", linewidth=2.2, label="LTP (Güçlenme: dt > 0)")
        ax1.plot(delta_t_neg, dw_neg, color="#c0392b", linewidth=2.2, label="LTD (Zayıflama: dt < 0)")
        ax1.axhline(0, color="#7f8c8d", linestyle="--", linewidth=1.0)
        ax1.axvline(0, color="#7f8c8d", linestyle="--", linewidth=1.0)
        ax1.set_title("1. Üstel STDP Plastisite Penceresi (Delta W)", fontsize=11, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("Spike Zaman Farkı Delta t = t_post - t_pre (ms)", fontsize=9)
        ax1.set_ylabel("Ağırlık Değişimi Delta W", fontsize=9)
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # ------------------------------------------------------------------
        # Panel 2: İlk vs Son Sinaptik Ağırlık Karşılaştırma Heatmap'i
        # ------------------------------------------------------------------
        ax2 = axes[0, 1]
        im2 = ax2.imshow(final_weights, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
        plt.colorbar(im2, ax=ax2, label="Sinaptik Ağırlık W")
        ax2.set_title("2. Son STDP Sinaptik Ağırlık Matrisi", fontsize=11, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("Presinaptik Girdi Nöronu", fontsize=9)
        ax2.set_ylabel("Postsinaptik Nöron", fontsize=9)

        # ------------------------------------------------------------------
        # Panel 3: İz (Trace) Zaman Serisi (x_pre(t) vs y_post(t))
        # ------------------------------------------------------------------
        ax3 = axes[0, 2]
        trace_pre = stdp_info["trace_pre"][0].cpu().numpy()  # (In,)
        trace_post = stdp_info["trace_post"][0].cpu().numpy()  # (Out,)
        
        ax3.bar(np.arange(len(trace_pre)), trace_pre, color="#2980b9", alpha=0.7, label="Presinaptik İz (x)")
        ax3.bar(np.arange(len(trace_post)), trace_post, color="#8e44ad", alpha=0.7, label="Postsinaptik İz (y)")
        ax3.set_title("3. Presinaptik & Postsinaptik İz (Trace) Akümülasyonu", fontsize=11, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Nöron İndeksi", fontsize=9)
        ax3.set_ylabel("İz Büyüklüğü", fontsize=9)
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, linestyle=":", alpha=0.6)

        # ------------------------------------------------------------------
        # Panel 4: Alıcı Alan (Receptive Field) Uzmanlaşma Ağırlıkları
        # ------------------------------------------------------------------
        ax4 = axes[1, 0]
        # İlk 4 postsinaptik nöronun ağırlık profili
        num_show = min(4, final_weights.shape[0])
        for i in range(num_show):
            ax4.plot(final_weights[i], label=f"Post Nöron {i+1}", linewidth=1.8, marker="o", markersize=3)
        ax4.set_title("4. Alıcı Alan (Receptive Field) Ağırlık Uzmanlaşması", fontsize=11, fontweight="bold", color="#2c3e50")
        ax4.set_xlabel("Girdi Özellik İndeksi", fontsize=9)
        ax4.set_ylabel("Öğrenilmiş Ağırlık W_ij", fontsize=9)
        ax4.legend(loc="upper right", fontsize=8)
        ax4.grid(True, linestyle=":", alpha=0.6)

        # ------------------------------------------------------------------
        # Panel 5: WTA Yanal İnhibisyon ve Spike Rekabeti
        # ------------------------------------------------------------------
        ax5 = axes[1, 1]
        spikes_seq = stdp_info["spikes_seq"][0].cpu().numpy()  # (T, Out)
        t_indices, neuron_indices = np.where(spikes_seq > 0.5)
        ax5.scatter(t_indices, neuron_indices, color="#d35400", s=25, marker="s", alpha=0.85)
        ax5.set_title("5. WTA Yanal İnhibisyon Rekabet Çıktısı", fontsize=11, fontweight="bold", color="#2c3e50")
        ax5.set_xlabel("Zaman Adımı (t)", fontsize=9)
        ax5.set_ylabel("Kazanan Postsinaptik Nöron", fontsize=9)
        ax5.set_ylim(-0.5, final_weights.shape[0] - 0.5)
        ax5.grid(True, linestyle=":", alpha=0.6)

        # ------------------------------------------------------------------
        # Panel 6: STDP Plastisite Performansı ve Entropi Dağılımı
        # ------------------------------------------------------------------
        ax6 = axes[1, 2]
        w_init_flat = initial_weights.flatten()
        w_final_flat = final_weights.flatten()
        
        ax6.hist(w_init_flat, bins=15, alpha=0.5, color="#7f8c8d", label="Başlangıç Ağırlıkları", density=True)
        ax6.hist(w_final_flat, bins=15, alpha=0.75, color="#27ae60", label="STDP Sonrası Ağırlıklar", density=True)
        
        ax6.set_title("6. Sinaptik Ağırlık Kutupsallaşması (Bimodal Bimodality)", fontsize=11, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Ağırlık Değeri W", fontsize=9)
        ax6.set_ylabel("Olasılık Yoğunluğu", fontsize=9)
        ax6.legend(loc="upper center", fontsize=8)
        ax6.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
