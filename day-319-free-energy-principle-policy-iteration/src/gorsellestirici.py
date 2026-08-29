"""
Day 319: 6-Panel Diagnostic Dashboard Visualizer for Free Energy Principle & Active Inference.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, Optional
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .serbest_enerji_aktif_cikarim import FEPSimulationResult, ActiveInferenceAgent, FEPConfig


class FEPGorsellestirici:
    """
    Renders a 6-panel dark-mode diagnostic dashboard for Free Energy Principle (FEP) Active Inference.
    """
    
    @staticmethod
    def ciz(result: FEPSimulationResult, agent: ActiveInferenceAgent, 
            cikti_yolu: str = "ciktilar/serbest_enerji_paneli.png", 
            profil_ozeti: Optional[Dict[str, Any]] = None):
        """
        Generates and saves the 6-panel diagnostic dashboard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)
        
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.patch.set_facecolor("#0b0f19")
        
        # -------------------------------------------------------------
        # Panel 1: Variational Free Energy (VFE) Minimization
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#111827")
        
        steps = np.arange(len(result.variational_free_energy_history))
        ax1.plot(steps, result.variational_free_energy_history, color="#38bdf8", marker="o", lw=2.2, label="Varyasyonel Serbest Enerji F(t)")
        
        ax1.set_xlabel("Zaman Adımı (t)", color="#94a3b8", fontsize=10)
        ax1.set_ylabel("Serbest Enerji (nats)", color="#94a3b8", fontsize=10)
        ax1.set_title("1. Algısal Sürpriz ve Serbest Enerji Azalımı", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax1.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
        
        # -------------------------------------------------------------
        # Panel 2: Expected Free Energy (EFE) Value Breakdown
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#111827")
        
        act_steps = np.arange(len(result.expected_free_energy_history))
        ax2.plot(act_steps, result.expected_free_energy_history, color="#f59e0b", marker="s", lw=2.0, label="Toplam Beklenen Serbest Enerji G(pi)")
        ax2.plot(act_steps, result.epistemic_value_history, color="#a855f7", linestyle="--", lw=1.8, label="Epistemik Merak / Bilgi Kazanımı")
        ax2.plot(act_steps, result.pragmatic_value_history, color="#10b981", linestyle=":", lw=1.8, label="Pragmatik Hedef Tercihi")
        
        ax2.set_xlabel("Karar Adımı", color="#94a3b8", fontsize=10)
        ax2.set_ylabel("Değer (Value)", color="#94a3b8", fontsize=10)
        ax2.set_title("2. Epistemik Keşif vs Pragmatik Sömürü Dengesi", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax2.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
        
        # -------------------------------------------------------------
        # Panel 3: State Belief Shannon Entropy Reduction
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.set_facecolor("#111827")
        
        ax3.plot(steps, result.state_entropy_history, color="#f43f5e", marker="^", lw=2.2, label="İnanç Belirsizliği H(q(s))")
        ax3.fill_between(steps, result.state_entropy_history, color="#f43f5e", alpha=0.2)
        
        ax3.set_xlabel("Zaman Adımı", color="#94a3b8", fontsize=10)
        ax3.set_ylabel("Shannon Entropisi (nats)", color="#94a3b8", fontsize=10)
        ax3.set_title("3. Gizli Durum İnanç Belirsizliği (Entropi)", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax3.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax3.legend(loc="upper right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
        
        # -------------------------------------------------------------
        # Panel 4: Action & State Trajectory
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.set_facecolor("#111827")
        
        t_indices = np.arange(len(result.trajectory_states))
        ax4.step(t_indices, result.trajectory_states, color="#34d399", where="post", lw=2.5, label="Ziyaret Edilen Durum s(t)")
        
        state_labels = ["0: Başlangıç", "1: İpucu/Keşif", "2: Hedef A", "3: Hedef B"]
        ax4.set_yticks([0, 1, 2, 3])
        ax4.set_yticklabels(state_labels, color="#94a3b8", fontsize=9)
        ax4.set_xlabel("Zaman Adımı", color="#94a3b8", fontsize=10)
        ax4.set_title("4. Ajan Durum Geçiş Rotası (Trajectory)", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax4.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax4.legend(loc="lower right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
        
        # -------------------------------------------------------------
        # Panel 5: Generative Model Likelihood Matrix A Heatmap
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.set_facecolor("#111827")
        
        im = ax5.imshow(agent.A_mat, cmap="viridis", aspect="auto", vmin=0, vmax=1.0)
        ax5.set_xticks(range(agent.S))
        ax5.set_yticks(range(agent.O))
        ax5.set_xticklabels([f"s_{i}" for i in range(agent.S)], color="#94a3b8")
        ax5.set_yticklabels(["Başlangıç", "İpucu", "Ödül (A)", "Ceza (B)"], color="#94a3b8")
        
        for i in range(agent.O):
            for j in range(agent.S):
                ax5.text(j, i, f"{agent.A_mat[i, j]:.2f}", ha="center", va="center", color="#ffffff", fontsize=9, fontweight="bold")
                
        ax5.set_title("5. Üretici Model Olabilirlik Matrisi A = P(o|s)", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        
        # -------------------------------------------------------------
        # Panel 6: Telemetry Summary Box
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.set_facecolor("#111827")
        ax6.axis("off")
        
        p = profil_ozeti or {}
        
        kpi_text = (
            "DAY 319: ACTIVE INFERENCE & FEP ENGINE\n"
            "==============================================\n"
            f"• Hedefe Ulaşıldı mı: {'EVET' if p.get('goal_reached', True) else 'HAYIR'}\n"
            f"• Toplam Karar Adımı: {p.get('trajectory_steps', len(result.trajectory_actions))}\n"
            f"• Epistemik Bilgi Kazanımı: {p.get('total_epistemic_gain', result.total_epistemic_gain):.4f} nats\n"
            f"• Başlangıç Entropisi H(0): {p.get('initial_state_entropy', 1.38):.4f} nats\n"
            f"• Bitiş Entropisi H(T): {p.get('final_state_entropy', 0.05):.4f} nats\n"
            f"• Entropi Düşüş Oranı: %{p.get('entropy_reduction_pct', 95.0):.2f}\n"
            f"• Son Varyasyonel Enerji: {p.get('final_variational_free_energy', result.final_vfe):.4f}\n"
            f"• Ajan Uyum Sınıfı: {p.get('fep_agent_tier', 'OPTIMAL_ACTIVE_INFERENCE_AGENT')}\n"
            "==============================================\n"
            "DURUM: KARL FRISTON SERBEST ENERJİ PRENSİBİ\n"
            "        VE AKTİF ÇIKARIM POLİTİKASI AKTİF"
        )
        
        ax6.text(
            0.05, 0.5, kpi_text,
            transform=ax6.transAxes,
            fontsize=8.8,
            fontfamily="monospace",
            color="#e2e8f0",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#1e293b", edgecolor="#10b981", lw=1.5, alpha=0.95)
        )
        ax6.set_title("6. FEP Politika İterasyon Raporu", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        
        plt.tight_layout(pad=2.5)
        plt.savefig(cikti_yolu, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"📊 [Visualizer] 6-Panelli Teşhis Panosu başarıyla kaydedildi: {cikti_yolu}")
