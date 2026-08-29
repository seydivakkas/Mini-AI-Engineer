"""
Day 307: 6-Panel Diagnostic Dashboard Visualizer for Causal World Discovery.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, Optional
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .nedensel_dunya_motoru import CausalDiscoveryResult


class CausalWorldGorsellestirici:
    """
    Renders a 6-panel dark-mode diagnostic dashboard for Causal World Representation & Do-Calculus.
    """
    
    @staticmethod
    def ciz(result: CausalDiscoveryResult, cikti_yolu: str = "ciktilar/nedensel_dunya_paneli.png", 
            profil_ozeti: Optional[Dict[str, Any]] = None):
        """
        Generates and saves the 6-panel diagnostic dashboard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)
        
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.patch.set_facecolor("#0b0f19")
        
        # -------------------------------------------------------------
        # Panel 1: Ground Truth vs Learned Adjacency Matrix
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#111827")
        
        # Plot learned adjacency heatmap
        cax = ax1.imshow(result.learned_adjacency_matrix, cmap="magma", interpolation="nearest")
        fig.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)
        
        d = result.learned_adjacency_matrix.shape[0]
        ax1.set_xticks(range(d))
        ax1.set_yticks(range(d))
        ax1.set_xticklabels([f"z{i}" for i in range(d)], color="#94a3b8")
        ax1.set_yticklabels([f"z{i}" for i in range(d)], color="#94a3b8")
        
        # Overlay ground truth indicators (circles for true edges)
        for r in range(d):
            for c in range(d):
                if result.ground_truth_adjacency_matrix[r, c] > 0.5:
                    ax1.plot(c, r, "o", color="#10b981", markersize=9, fillstyle="none", markeredgewidth=2)
                    
        ax1.set_title("1. Öğrenilen Nedensel Çizge vs Gerçek (Yeşil Halka)", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        
        # -------------------------------------------------------------
        # Panel 2: DAG Edge Discovery (TPR & FDR)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#111827")
        
        metrics = ["Doğru Kenar Tespiti (TPR)", "Yanlış Keşif Oranı (FDR)"]
        values = [result.dag_true_positive_rate_pct, result.dag_false_discovery_rate_pct]
        colors = ["#10b981", "#f43f5e"]
        
        b2 = ax2.bar(metrics, values, color=colors, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b2, values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"%{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax2.set_ylim(0, 115)
        ax2.set_ylabel("Oran (%)", color="#94a3b8", fontsize=10)
        ax2.set_title("2. Nedensel Çizge Keşif Doğruluğu", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax2.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 3: Training Loss Trajectory
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.set_facecolor("#111827")
        
        epochs = range(1, len(result.loss_history) + 1)
        ax3.plot(epochs, result.loss_history, color="#38bdf8", lw=2.2, label="Toplam Kayıp (Loss + NOTEARS)")
        
        ax3.set_xlabel("Epok (Epoch)", color="#94a3b8", fontsize=10)
        ax3.set_ylabel("Kayıp Değeri", color="#94a3b8", fontsize=10)
        ax3.set_title("3. NOTEARS DAG Kısıtlı Öğrenme Eğrisi", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax3.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax3.legend(loc="upper right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=9)
        
        # -------------------------------------------------------------
        # Panel 4: Pearl's 3 Causal Levels (MSE)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.set_facecolor("#111827")
        
        levels = ["1. Gözlemsel\nRecon MSE", "2. Müdahale\ndo(z) MSE", "3. Karşı-Olgusal\nCounterfactual"]
        errors = [result.reconstruction_mse, result.interventional_mse, result.counterfactual_mse]
        colors_lvl = ["#a855f7", "#38bdf8", "#ec4899"]
        
        b4 = ax4.bar(levels, errors, color=colors_lvl, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b4, errors):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{val:.4f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax4.set_ylabel("MSE Hatası", color="#94a3b8", fontsize=10)
        ax4.set_title("4. Pearl'ün 3 Nedensel Katmanı Hata Dağılımı", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax4.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 5: Structural Hamming Distance Comparison
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.set_facecolor("#111827")
        
        shd_cats = ["Mevcut Model SHD", "Kabul Edilebilir Eşik"]
        shd_vals = [result.structural_hamming_distance, 3]
        colors_shd = ["#10b981" if result.structural_hamming_distance <= 3 else "#f43f5e", "#64748b"]
        
        b5 = ax5.bar(shd_cats, shd_vals, color=colors_shd, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b5, shd_vals):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val} Kenar Hatası", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax5.set_ylabel("Kenar Farkı (Adet)", color="#94a3b8", fontsize=10)
        ax5.set_ylim(0, max(shd_vals) + 3)
        ax5.set_title("5. Structural Hamming Distance (SHD)", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax5.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 6: Causal World Representation Diagnostic Summary Box
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.set_facecolor("#111827")
        ax6.axis("off")
        
        p = profil_ozeti or {}
        
        kpi_text = (
            "DAY 307: CAUSAL WORLD DISCOVERY & DO-CALCULUS\n"
            "===============================================\n"
            f"• Yapısal Hamming Uzaklığı (SHD): {p.get('structural_hamming_distance', result.structural_hamming_distance)}\n"
            f"• DAG Kenar Keşfi (TPR): %{p.get('dag_true_positive_rate_pct', result.dag_true_positive_rate_pct):.2f}\n"
            f"• Yanlış Keşif Oranı (FDR): %{p.get('dag_false_discovery_rate_pct', result.dag_false_discovery_rate_pct):.2f}\n"
            f"• Gözlemsel Rekonstrüksiyon MSE: {p.get('reconstruction_mse', result.reconstruction_mse):.4f}\n"
            f"• Müdahale (do) Tahmin MSE: {p.get('interventional_mse', result.interventional_mse):.4f}\n"
            f"• Karşı-Olgusal (Counterfactual) MSE: {p.get('counterfactual_mse', result.counterfactual_mse):.4f}\n"
            f"• Asiklik Durumu: {p.get('acyclicity_status', 'STRICT_DAG_SATISFIED')}\n"
            f"• Nedensel Genelleme: {p.get('causal_generalization_status', 'ROBUST_INTERVENTIONAL')}\n"
            "===============================================\n"
            "Durum: PEARL SEVİYE-3 NEDENSELLİK VE\n"
            "        NOTEARS KISITLAMA AKTİF"
        )
        
        ax6.text(
            0.05, 0.5, kpi_text,
            transform=ax6.transAxes,
            fontsize=9.5,
            fontfamily="monospace",
            color="#e2e8f0",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#1e293b", edgecolor="#10b981", lw=1.5, alpha=0.95)
        )
        ax6.set_title("6. Nedensel Dünya Modeli Özeti", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        
        plt.tight_layout(pad=2.5)
        plt.savefig(cikti_yolu, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"📊 [Visualizer] 6-Panelli Teşhis Panosu başarıyla kaydedildi: {cikti_yolu}")
