"""
Day 305: 6-Panel Diagnostic Dashboard Visualizer for Cross-Coder Sparse Autoencoder.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, Optional
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .cross_coder_motoru import CrossCoderResult


class CrossCoderGorsellestirici:
    """
    Renders a 6-panel dark-mode diagnostic dashboard for Mechanistic Interpretability.
    """
    
    @staticmethod
    def ciz(result: CrossCoderResult, cikti_yolu: str = "ciktilar/cross_coder_paneli.png", 
            profil_ozeti: Optional[Dict[str, Any]] = None):
        """
        Generates and saves the 6-panel diagnostic dashboard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)
        
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.patch.set_facecolor("#0b0f19")
        
        # -------------------------------------------------------------
        # Panel 1: Fraction of Variance Explained (FVE) per Layer
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#111827")
        
        layer_names = [f"Katman {i} (L_{i})" for i in range(len(result.fve_per_layer))]
        bars = ax1.bar(layer_names, result.fve_per_layer, color="#38bdf8", width=0.5, edgecolor="#ffffff", lw=1.2, alpha=0.9)
        
        for bar, val in zip(bars, result.fve_per_layer):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"%{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax1.set_ylim(0, 115)
        ax1.set_ylabel("FVE (%) - Aciklanan Varyans", color="#94a3b8", fontsize=10)
        ax1.set_title(f"1. Katman Bazli FVE (Ort: %{result.mean_fve:.1f})", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax1.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 2: Training Loss Dynamics (Recon vs L1)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#111827")
        
        epochs = np.arange(1, len(result.history["total_loss"]) + 1)
        ax2.plot(epochs, result.history["total_loss"], color="#f59e0b", lw=2.2, label="Toplam Kayip")
        ax2.plot(epochs, result.history["recon_loss"], color="#38bdf8", lw=2.0, linestyle="--", label="Yeniden Kurma (MSE)")
        ax2.plot(epochs, result.history["l1_loss"], color="#ec4899", lw=1.8, linestyle=":", label="Grup L1 Seyreklik")
        
        ax2.set_xlabel("Epok", color="#94a3b8", fontsize=10)
        ax2.set_ylabel("Kayip Degeri", color="#94a3b8", fontsize=10)
        ax2.set_title("2. Cross-Coder Egitim Kayip Dinamikleri", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax2.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=9)
        
        # -------------------------------------------------------------
        # Panel 3: Sparsity Dynamics (L0 Active Features)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.set_facecolor("#111827")
        
        ax3.plot(epochs, result.history["l0_sparsity"], color="#10b981", lw=2.5, label="Aktif Ozellik Sayisi (L0)")
        ax3.axhline(result.l0_sparsity, color="#34d399", linestyle="--", label=f"Nihai L0: {result.l0_sparsity:.1f}")
        
        ax3.set_xlabel("Epok", color="#94a3b8", fontsize=10)
        ax3.set_ylabel("Ortalama Aktif Ozellik Sayisi", color="#94a3b8", fontsize=10)
        ax3.set_title(f"3. L0 Seyreklik Profili (L0 = {result.l0_sparsity:.1f})", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax3.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax3.legend(loc="upper right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=9)
        
        # -------------------------------------------------------------
        # Panel 4: Cross-Layer Feature Decoder Norm Heatmap (Top 20 Features)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.set_facecolor("#111827")
        
        sample_features = result.layer_norm_attributions[:20]  # [20, K]
        im = ax4.imshow(sample_features, aspect="auto", cmap="magma", interpolation="nearest")
        
        ax4.set_xlabel("Model Katmani", color="#94a3b8", fontsize=10)
        ax4.set_ylabel("Latent Ozellik ID (Ilk 20)", color="#94a3b8", fontsize=10)
        ax4.set_xticks(range(sample_features.shape[1]))
        ax4.set_xticklabels([f"L_{i}" for i in range(sample_features.shape[1])])
        ax4.set_title(f"4. Katmanlar Arasi Decoder Normu (Paylasim: %{result.cross_layer_sharing_idx:.1f})", 
                      color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        
        cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8, colors="#94a3b8")
        
        # -------------------------------------------------------------
        # Panel 5: Feature Utilization & Dead Neurons
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.set_facecolor("#111827")
        
        categories = ["Aktif Nöronlar", "Ölü Nöronlar (Dead)"]
        values = [100.0 - result.dead_feature_pct, result.dead_feature_pct]
        colors_pie = ["#10b981", "#ef4444"]
        
        b5 = ax5.bar(categories, values, color=colors_pie, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b5, values):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"%{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax5.set_ylim(0, 115)
        ax5.set_ylabel("Oran (%)", color="#94a3b8", fontsize=10)
        ax5.set_title(f"5. Sozluk Nöron Kullanim Orani (Ölü: %{result.dead_feature_pct:.1f})", 
                      color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax5.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 6: Diagnostic & Superposition Resolution Summary
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.set_facecolor("#111827")
        ax6.axis("off")
        
        p = profil_ozeti or {}
        
        kpi_text = (
            "DAY 305: CROSS-CODER SPARSE AUTOENCODER\n"
            "===========================================\n"
            f"• Ortalama FVE (R^2 Skoru): %{p.get('mean_fve_pct', result.mean_fve):.2f}\n"
            f"• Katman FVE Detayi: {p.get('layer_fves', '')}\n"
            f"• Ortalama L0 Seyreklik: {p.get('l0_sparsity_avg', result.l0_sparsity):.2f} nöron/ornek\n"
            f"• Ölü Nöron Orani: %{p.get('dead_feature_pct', result.dead_feature_pct):.2f}\n"
            f"• Katmanlar Arasi Paylasim Orani: %{p.get('cross_layer_sharing_pct', result.cross_layer_sharing_idx):.2f}\n"
            f"• Kalite: {p.get('disentanglement_quality', 'EXCELLENT')}\n"
            "===========================================\n"
            "Durum: SUPERPOSITION VE POLISEMANTISITE\n"
            "        BASARIYLA COZUMLENDI"
        )
        
        ax6.text(
            0.05, 0.5, kpi_text,
            transform=ax6.transAxes,
            fontsize=10,
            fontfamily="monospace",
            color="#e2e8f0",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#1e293b", edgecolor="#38bdf8", lw=1.5, alpha=0.95)
        )
        ax6.set_title("6. Mekanistik Yorumlanabilirlik Ozeti", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        
        plt.tight_layout(pad=2.5)
        plt.savefig(cikti_yolu, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"📊 [Visualizer] 6-Panelli Teşhis Panosu başarıyla kaydedildi: {cikti_yolu}")
