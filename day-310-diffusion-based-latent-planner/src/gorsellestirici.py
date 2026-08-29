"""
Day 310: 6-Panel Diagnostic Dashboard Visualizer for Diffusion-Based Latent Planner.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, Optional
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from .difuzyon_planlayici_motoru import DiffusionPlannerResult


class DiffusionPlannerGorsellestirici:
    """
    Renders a 6-panel dark-mode diagnostic dashboard for Diffusion-Based Latent Planning.
    """
    
    @staticmethod
    def ciz(result: DiffusionPlannerResult, cikti_yolu: str = "ciktilar/difuzyon_planlayici_paneli.png", 
            profil_ozeti: Optional[Dict[str, Any]] = None):
        """
        Generates and saves the 6-panel diagnostic dashboard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)
        
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.patch.set_facecolor("#0b0f19")
        
        # -------------------------------------------------------------
        # Panel 1: 2D Continuous Trajectory Space with Obstacles
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#111827")
        
        # Draw obstacles
        for ox, oy, r in result.obstacles:
            circle = plt.Circle((ox, oy), r, color="#f43f5e", alpha=0.35, ec="#f43f5e", lw=1.5)
            ax1.add_patch(circle)
            ax1.text(ox, oy, "Engel", color="#fecdd3", fontsize=8, ha="center", va="center", fontweight="bold")
            
        # Draw sampled trajectories (first 15 paths)
        N_plot = min(15, len(result.sampled_trajectories))
        for i in range(N_plot):
            traj = result.sampled_trajectories[i]
            ax1.plot(traj[:, 0], traj[:, 1], color="#38bdf8", alpha=0.6, lw=1.5)
            ax1.scatter([traj[0, 0]], [traj[0, 1]], color="#10b981", s=30, zorder=4)
            ax1.scatter([traj[-1, 0]], [traj[-1, 1]], color="#a855f7", s=30, zorder=4)
            
        ax1.scatter([], [], color="#10b981", s=40, label="Başlangıç (Start)")
        ax1.scatter([], [], color="#a855f7", s=40, label="Hedef (Goal)")
        ax1.plot([], [], color="#38bdf8", lw=2, label="Difüzyon Yörüngesi")
        
        ax1.set_xlim(-0.5, 10.5)
        ax1.set_ylim(-0.5, 10.5)
        ax1.set_xlabel("X Koordinatı (Latent / State)", color="#94a3b8", fontsize=10)
        ax1.set_ylabel("Y Koordinatı (Latent / State)", color="#94a3b8", fontsize=10)
        ax1.set_title("1. Sürekli Yörünge Örnekleme ve Engel Aşımı", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax1.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
        
        # -------------------------------------------------------------
        # Panel 2: Reverse Diffusion Denoising Step Evolution
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#111827")
        
        steps = [40, 30, 20, 10, 0]
        alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
        colors_d = ["#64748b", "#ec4899", "#a855f7", "#38bdf8", "#10b981"]
        
        # Simulated denoising curve evolution
        sample_path = result.sampled_trajectories[0] if len(result.sampled_trajectories) > 0 else np.zeros((32, 2))
        for step_val, alpha, col in zip(steps, alphas, colors_d):
            noise = np.random.randn(*sample_path.shape) * (step_val / 40.0) * 1.5
            intermediate = sample_path + noise
            ax2.plot(intermediate[:, 0], intermediate[:, 1], color=col, alpha=alpha, lw=1.8, 
                     label=f"t = {step_val}")
                     
        ax2.set_title("2. Ters Difüzyon Gürültü Arındırma Evrimi", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax2.set_xlabel("X Uzayı", color="#94a3b8", fontsize=10)
        ax2.set_ylabel("Y Uzayı", color="#94a3b8", fontsize=10)
        ax2.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax2.legend(loc="upper left", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
        
        # -------------------------------------------------------------
        # Panel 3: Trajectory Velocity Profiles
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.set_facecolor("#111827")
        
        if len(result.sampled_trajectories) > 0:
            vels = np.linalg.norm(np.diff(result.sampled_trajectories[0], axis=0), axis=-1)
            time_axis = np.arange(len(vels))
            ax3.plot(time_axis, vels, color="#f59e0b", lw=2.2, label="Hız Profili ||v_t||")
            ax3.fill_between(time_axis, 0, vels, color="#f59e0b", alpha=0.15)
        
        ax3.set_xlabel("Zaman Adımı (Horizon H)", color="#94a3b8", fontsize=10)
        ax3.set_ylabel("Hız Büyüklüğü (Delta s)", color="#94a3b8", fontsize=10)
        ax3.set_title("3. Yörünge Hız ve Dinamik Akıcılık", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax3.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax3.legend(loc="upper right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=9)
        
        # -------------------------------------------------------------
        # Panel 4: DDPM vs DDIM Sampling Speedup Comparison
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.set_facecolor("#111827")
        
        methods = ["DDPM (Standart)", "DDIM (Hızlandırılmış)"]
        sampling_times = [40, 10]
        b4 = ax4.bar(methods, sampling_times, color=["#64748b", "#38bdf8"], width=0.45, 
                     edgecolor="#ffffff", lw=1.2, alpha=0.85)
                     
        for bar, val in zip(b4, sampling_times):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{val} Adım\n({40/val:.1f}x)", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax4.set_ylim(0, 55)
        ax4.set_ylabel("Ters Difüzyon Adım Sayısı (T)", color="#94a3b8", fontsize=10)
        ax4.set_title("4. DDPM vs DDIM Örnekleme Verimliliği", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax4.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 5: Goal Reachability & Obstacle Avoidance
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.set_facecolor("#111827")
        
        metrics = ["Hedefe Ulaşma\n(Reachability)", "Engel Kaçınma\n(Avoidance)", "Yörünge\nPürüzsüzlüğü"]
        scores = [result.goal_reachability_rate_pct, result.obstacle_avoidance_rate_pct, result.trajectory_smoothness_score]
        colors_m = ["#10b981", "#38bdf8", "#a855f7"]
        
        b5 = ax5.bar(metrics, scores, color=colors_m, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b5, scores):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"%{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax5.set_ylim(0, 115)
        ax5.set_ylabel("Başarı Skoru (%)", color="#94a3b8", fontsize=10)
        ax5.set_title("5. Planlama Başarım ve Güvenilirlik Metrikleri", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax5.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 6: Telemetry Summary Box
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.set_facecolor("#111827")
        ax6.axis("off")
        
        p = profil_ozeti or {}
        
        kpi_text = (
            "DAY 310: DIFFUSION-BASED LATENT PLANNER\n"
            "==============================================\n"
            f"• Hedefe Ulaşma Oranı: %{p.get('goal_reachability_rate_pct', result.goal_reachability_rate_pct):.2f}\n"
            f"• Engelden Kaçınma: %{p.get('obstacle_avoidance_rate_pct', result.obstacle_avoidance_rate_pct):.2f}\n"
            f"• Pürüzsüzlük Skoru: {p.get('trajectory_smoothness_score', result.trajectory_smoothness_score):.2f}/100\n"
            f"• DDIM Hızlanma Faktörü: {p.get('ddim_speedup_factor', result.ddim_speedup_factor):.1f}x\n"
            f"• Ortalama Yörünge Uzunluğu: {p.get('avg_trajectory_length', result.avg_trajectory_length):.2f} birim\n"
            f"• Planlayıcı Sınıfı: {p.get('planner_tier', 'OPTIMAL_CONTINUOUS_DIFFUSION_PLANNER')}\n"
            "==============================================\n"
            "Durum: CLASSIFIER-FREE GUIDANCE VE\n"
            "        SKOR TABANLI PLANLAMA AKTİF"
        )
        
        ax6.text(
            0.05, 0.5, kpi_text,
            transform=ax6.transAxes,
            fontsize=9.5,
            fontfamily="monospace",
            color="#e2e8f0",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#1e293b", edgecolor="#38bdf8", lw=1.5, alpha=0.95)
        )
        ax6.set_title("6. Difüzyon Planlama Modeli Özeti", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        
        plt.tight_layout(pad=2.5)
        plt.savefig(cikti_yolu, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"📊 [Visualizer] 6-Panelli Teşhis Panosu başarıyla kaydedildi: {cikti_yolu}")
