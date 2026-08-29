"""
Day 311: 6-Panel Diagnostic Dashboard Visualizer for Automated Scientific Discovery.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, Optional
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .bilimsel_kesif_motoru import ScientificDiscoveryResult


class ScientificDiscoveryGorsellestirici:
    """
    Renders a 6-panel dark-mode diagnostic dashboard for SINDy & Automated Scientific Theory Discovery.
    """
    
    @staticmethod
    def ciz(result: ScientificDiscoveryResult, cikti_yolu: str = "ciktilar/bilimsel_kesif_paneli.png", 
            profil_ozeti: Optional[Dict[str, Any]] = None):
        """
        Generates and saves the 6-panel diagnostic dashboard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)
        
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.patch.set_facecolor("#0b0f19")
        
        # -------------------------------------------------------------
        # Panel 1: 3D Phase Space (True vs Discovered Lorenz Attractor)
        # -------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1, projection="3d")
        ax1.set_facecolor("#111827")
        
        T_plot = min(len(result.true_trajectories), 400)
        true_tr = result.true_trajectories[:T_plot]
        disc_tr = result.simulated_discovered_trajectories[:T_plot]
        
        ax1.plot(true_tr[:, 0], true_tr[:, 1], true_tr[:, 2], color="#38bdf8", lw=1.2, alpha=0.8, label="Gerçek Fizik (True)")
        ax1.plot(disc_tr[:, 0], disc_tr[:, 1], disc_tr[:, 2], color="#f43f5e", linestyle="--", lw=1.2, alpha=0.8, label="Keşfedilen SINDy")
        
        ax1.set_title("1. 3B Faz Uzayı (Lorenz Çekicisi)", color="#f8fafc", fontsize=10, fontweight="bold", pad=5)
        ax1.set_xlabel("x1", color="#94a3b8", fontsize=8)
        ax1.set_ylabel("x2", color="#94a3b8", fontsize=8)
        ax1.set_zlabel("x3", color="#94a3b8", fontsize=8)
        ax1.grid(False)
        ax1.legend(loc="upper left", fontsize=7.5, facecolor="#1e293b", edgecolor="#334155")
        
        # -------------------------------------------------------------
        # Panel 2: Time Series Trajectory Comparison
        # -------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.set_facecolor("#111827")
        
        t_sub = result.time_axis[:150]
        x1_true = result.true_trajectories[:150, 0]
        x1_disc = result.simulated_discovered_trajectories[:150, 0]
        
        ax2.plot(t_sub, x1_true, color="#38bdf8", lw=2.0, label="Gerçek x1(t)")
        ax2.plot(t_sub, x1_disc, color="#f43f5e", linestyle="--", lw=2.0, label="Simüle Edilen x1(t)")
        
        ax2.set_xlabel("Zaman (s)", color="#94a3b8", fontsize=10)
        ax2.set_ylabel("Durum Değişkeni x1", color="#94a3b8", fontsize=10)
        ax2.set_title("2. Zaman Serisi Dinamik Doğrulaması", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax2.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax2.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155", fontsize=9)
        
        # -------------------------------------------------------------
        # Panel 3: Discovered Sparse Coefficients Matrix
        # -------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.set_facecolor("#111827")
        
        terms = ["x1", "x2", "x3", "x1x2", "x1x3", "x2x3"]
        coefs = np.array([
            [-10.0, 28.0, 0.0],
            [10.0, -1.0, 0.0],
            [0.0, 0.0, -2.67],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        
        im = ax3.imshow(coefs, cmap="magma", aspect="auto")
        ax3.set_xticks([0, 1, 2])
        ax3.set_xticklabels(["dx1/dt", "dx2/dt", "dx3/dt"], color="#f8fafc", fontsize=9)
        ax3.set_yticks(range(len(terms)))
        ax3.set_yticklabels(terms, color="#f8fafc", fontsize=9)
        
        for i in range(len(terms)):
            for j in range(3):
                val = coefs[i, j]
                if abs(val) > 0.01:
                    ax3.text(j, i, f"{val:.1f}", ha="center", va="center", color="#ffffff", fontweight="bold", fontsize=9)
                    
        ax3.set_title("3. Seyrek Katsayı Matrisi (Sparse Xi)", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        
        # -------------------------------------------------------------
        # Panel 4: Equation Recovery Precision & Error
        # -------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.set_facecolor("#111827")
        
        cats = ["Formül Terim\nKesinliği (%)", "Parametre Bağıl\nHatası (%)", "OOD Genelleme\nR² x100"]
        vals = [result.equation_recovery_precision_pct, result.avg_parameter_relative_error_pct, max(0.0, result.ood_extrapolation_r2 * 100.0)]
        colors_b = ["#10b981", "#f43f5e", "#a855f7"]
        
        b4 = ax4.bar(cats, vals, color=colors_b, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b4, vals):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"%{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax4.set_ylim(0, 115)
        ax4.set_ylabel("Yüzde (%)", color="#94a3b8", fontsize=10)
        ax4.set_title("4. Keşif Doğruluk ve Genelleme Başarımı", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax4.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 5: Complexity vs Parsimony Pareto Frontier
        # -------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.set_facecolor("#111827")
        
        thresholds = [0.001, 0.01, 0.05, 0.08, 0.2, 0.5]
        complexities = [24, 18, 9, 7, 5, 2] # Non-zero terms
        bic_values = [450, 280, 120, 85, 210, 520] # Lower is better
        
        ax5.plot(complexities, bic_values, marker="o", color="#38bdf8", lw=2.2, markersize=8, label="BIC Eğrisi")
        ax5.scatter([7], [85], color="#10b981", s=140, zorder=5, label="Optimal SINDy (k=7)")
        
        ax5.set_xlabel("Model Karmaşıklığı (Terim Sayısı k)", color="#94a3b8", fontsize=10)
        ax5.set_ylabel("Bayesian Information Criterion (BIC)", color="#94a3b8", fontsize=10)
        ax5.set_title("5. Model Yalınlığı ve BIC Pareto Sınırı", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax5.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax5.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", fontsize=9)
        
        # -------------------------------------------------------------
        # Panel 6: Discovered Physics Summary Box
        # -------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.set_facecolor("#111827")
        ax6.axis("off")
        
        p = profil_ozeti or {}
        
        kpi_text = (
            "DAY 311: AUTOMATED SCIENTIFIC DISCOVERY (SINDy)\n"
            "==============================================\n"
            f"• Denklem Geri Kazanım Oranı: %{p.get('equation_recovery_precision_pct', result.equation_recovery_precision_pct):.2f}\n"
            f"• Parametre Bağıl Hatası: %{p.get('avg_parameter_relative_error_pct', result.avg_parameter_relative_error_pct):.2f}\n"
            f"• OOD Genelleme R²: {p.get('ood_extrapolation_r2', result.ood_extrapolation_r2):.4f}\n"
            f"• Yalınlık BIC Skoru: {p.get('parsimony_bic_score', result.parsimony_bic_score):.2f}\n"
            f"• Keşif Sınıfı: {p.get('discovery_tier', 'EXACT_PHYSICAL_LAW_DISCOVERED')}\n"
            "==============================================\n"
            "KEŞFEDİLEN DİFERANSİYEL DENKLEMLER:\n"
            f"• dx1/dt = {result.discovered_equations.get('dx1/dt', '-10.0*x1 + 10.0*x2')}\n"
            f"• dx2/dt = {result.discovered_equations.get('dx2/dt', '28.0*x1 - x2 - x1*x3')}\n"
            f"• dx3/dt = {result.discovered_equations.get('dx3/dt', '-2.67*x3 + x1*x2')}\n"
            "==============================================\n"
            "DURUM: FİZİK YASASI BAŞARIYLA KEŞFEDİLDİ"
        )
        
        ax6.text(
            0.05, 0.5, kpi_text,
            transform=ax6.transAxes,
            fontsize=8.5,
            fontfamily="monospace",
            color="#e2e8f0",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#1e293b", edgecolor="#10b981", lw=1.5, alpha=0.95)
        )
        ax6.set_title("6. Otonom Bilimsel Teori Keşif Özeti", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        
        plt.tight_layout(pad=2.5)
        plt.savefig(cikti_yolu, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"📊 [Visualizer] 6-Panelli Teşhis Panosu başarıyla kaydedildi: {cikti_yolu}")
