"""
Day 303: 6-Panel Diagnostic Dashboard Visualizer for Quality-Diversity & POET.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, Optional
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .map_elites_poet_motoru import QDResult


class POETGorsellestirici:
    """
    Renders a 6-panel dark-mode diagnostic dashboard for MAP-Elites & POET co-evolution.
    """
    
    @staticmethod
    def ciz(result: QDResult, cikti_yolu: str = "ciktilar/poet_qd_paneli.png", 
            profil_ozeti: Optional[Dict[str, Any]] = None):
        """
        Generates and saves the 6-panel diagnostic dashboard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)
        
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.patch.set_facecolor("#0b0f19")
        
        # -------------------------------------------------------------
        # Panel 1: 2D MAP-Elites Behavioral Archive Heatmap
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#111827")
        
        grid = result.archive_grid.copy()
        # Mask unoccupied cells
        masked_grid = np.ma.masked_where(grid == -np.inf, grid)
        
        im1 = ax1.imshow(masked_grid, cmap="plasma", origin="lower", aspect="auto", vmin=0.0, vmax=100.0)
        ax1.set_title("1. MAP-Elites 2D Davranissal Arsiv Isisi", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax1.set_xlabel("Davranis b1 (Enerji / Hiz Profili)", color="#94a3b8", fontsize=10)
        ax1.set_ylabel("Davranis b2 (Kesif / Simetri)", color="#94a3b8", fontsize=10)
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label("Uygunluk (Fitness)", color="#94a3b8", fontsize=9)
        cbar1.ax.tick_params(colors="#94a3b8")
        
        # -------------------------------------------------------------
        # Panel 2: QD-Score and Archive Coverage Growth
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#111827")
        
        iters = np.arange(1, len(result.qd_score_history) + 1)
        color_qd = "#38bdf8"
        color_cov = "#10b981"
        
        ax2.plot(iters, result.qd_score_history, color=color_qd, lw=2.2, label="QD-Score")
        ax2.set_xlabel("Evrim Iterasyonu", color="#94a3b8", fontsize=10)
        ax2.set_ylabel("Toplam QD-Skoru", color=color_qd, fontsize=10)
        ax2.tick_params(axis="y", labelcolor=color_qd)
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(iters, result.coverage_history, color=color_cov, lw=2.2, linestyle="--", label="Kapsama (%)")
        ax2_twin.set_ylabel("Arsiv Kapsamasi (%)", color=color_cov, fontsize=10)
        ax2_twin.tick_params(axis="y", labelcolor=color_cov)
        
        ax2.set_title("2. QD-Skoru ve Arsiv Kapsama Gelisimi", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax2.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 3: POET Co-evolved Environmental Niches
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.set_facecolor("#111827")
        
        envs = result.active_envs
        if envs:
            roughness = [e.roughness for e in envs]
            gaps = [e.gap_width for e in envs]
            obstacles = [e.obstacle_density * 150 + 50 for e in envs]
            
            sc3 = ax3.scatter(roughness, gaps, s=obstacles, c=range(len(envs)), cmap="viridis", alpha=0.85, edgecolors="#38bdf8", lw=1.5)
            for i, env in enumerate(envs):
                ax3.annotate(f"E{env.env_id}", (env.roughness + 0.02, env.gap_width + 0.02), color="#e2e8f0", fontsize=9)
            cbar3 = fig.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.set_label("Ortam Nesli / ID", color="#94a3b8", fontsize=9)
            cbar3.ax.tick_params(colors="#94a3b8")
        else:
            ax3.text(0.5, 0.5, "Tekil Ortam Modu", ha="center", va="center", color="#94a3b8")
            
        ax3.set_title("3. POET Ortam Cukurlari (Niches)", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax3.set_xlabel("Zemin Pruzlulugu (Roughness)", color="#94a3b8", fontsize=10)
        ax3.set_ylabel("Bosluk Genisligi (Gap Width)", color="#94a3b8", fontsize=10)
        ax3.set_xlim(-0.05, 1.05)
        ax3.set_ylim(-0.05, 1.05)
        ax3.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 4: Direct Policy Transfer Matrix
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.set_facecolor("#111827")
        
        mat = result.transfer_matrix
        if mat.ndim == 2 and mat.size > 1 and mat.shape[0] > 1:
            im4 = ax4.imshow(mat, cmap="magma", aspect="auto", vmin=0.0, vmax=100.0)
            ax4.set_title("4. Capraz-Ortam Politika Transfer Matrisi", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
            ax4.set_xlabel("Hedef Ortam (Target Env)", color="#94a3b8", fontsize=10)
            ax4.set_ylabel("Kaynak Ajan (Source Agent)", color="#94a3b8", fontsize=10)
            ax4.set_xticks(range(mat.shape[1]))
            ax4.set_yticks(range(mat.shape[0]))
            cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
            cbar4.set_label("Transfer Uygunlugu", color="#94a3b8", fontsize=9)
            cbar4.ax.tick_params(colors="#94a3b8")
        else:
            ax4.text(0.5, 0.5, "Transfer Matrisi (N>=2)", ha="center", va="center", color="#94a3b8")
            ax4.set_title("4. Politika Transfer Matrisi", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
            
        # -------------------------------------------------------------
        # Panel 5: Elite Fitness Distribution Across Niches
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.set_facecolor("#111827")
        
        valid_fits = grid[grid > -np.inf]
        if len(valid_fits) > 0:
            ax5.hist(valid_fits, bins=20, color="#a855f7", edgecolor="#c084fc", alpha=0.75, rwidth=0.85)
            mean_f = np.mean(valid_fits)
            max_f = np.max(valid_fits)
            ax5.axvline(mean_f, color="#38bdf8", linestyle="--", lw=1.8, label=f"Ortalama ({mean_f:.1f})")
            ax5.axvline(max_f, color="#f43f5e", linestyle=":", lw=2.0, label=f"Zirve ({max_f:.1f})")
            
        ax5.set_title("5. Elit Birey Uygunluk Dagilimi", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax5.set_xlabel("Uygunluk Skoru (Fitness)", color="#94a3b8", fontsize=10)
        ax5.set_ylabel("Cukur (Niche) Frekansi", color="#94a3b8", fontsize=10)
        ax5.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax5.legend(loc="upper left", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=9)
        
        # -------------------------------------------------------------
        # Panel 6: Diagnostic & Quality-Diversity Summary KPI Box
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.set_facecolor("#111827")
        ax6.axis("off")
        
        p = profil_ozeti or {}
        best = result.best_individual
        
        kpi_text = (
            "DAY 303: MAP-ELITES & POET SUMMARY\n"
            "===========================================\n"
            f"• Toplam QD-Skoru: {p.get('qd_score', 0.0):.2f}\n"
            f"• Arsiv Kapsamasi: %{p.get('archive_coverage_pct', 0.0):.2f}\n"
            f"• Doldurulan Niche Sayisi: {p.get('total_occupied_niches', 0)} / {p.get('total_grid_capacity', 256)}\n"
            f"• En Yuksek Elit Uygunluk: {p.get('max_elite_fitness', best.fitness):.2f}\n"
            f"• Ortalama Elit Uygunluk: {p.get('mean_elite_fitness', 0.0):.2f}\n"
            f"• Aktif POET Ortam Sayisi: {p.get('active_environments_count', len(result.active_envs))}\n"
            f"• Ortalama Zemin Pruzlulugu: {p.get('avg_env_roughness', 0.0):.3f}\n"
            f"• Capraz-Transfer Basari Orani: %{p.get('cross_transfer_success_rate', 0.0):.1f}\n"
            f"• Toplam Evrimsel Degerlendirme: {p.get('total_evaluations', result.total_evaluations)}\n"
            "===========================================\n"
            f"[En Iyi Ajan]: ID #{best.ind_id} (Fit: {best.fitness:.2f})\n"
            "Durum: UCU ACIK KALITE-CESITLILIK BASARILI"
        )
        
        ax6.text(
            0.05, 0.5, kpi_text,
            transform=ax6.transAxes,
            fontsize=10,
            fontfamily="monospace",
            color="#e2e8f0",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#1e293b", edgecolor="#10b981", lw=1.5, alpha=0.95)
        )
        ax6.set_title("6. QD & POET Tehis Ozeti", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        
        plt.tight_layout(pad=2.5)
        plt.savefig(cikti_yolu, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"📊 [Visualizer] 6-Panelli Teşhis Panosu başarıyla kaydedildi: {cikti_yolu}")
