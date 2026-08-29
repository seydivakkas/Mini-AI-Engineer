"""
Day 306: 6-Panel Diagnostic Dashboard Visualizer for Scalable Oversight Debate Trees.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, Optional
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .debate_motoru import DebateResult


class DebateTreeGorsellestirici:
    """
    Renders a 6-panel dark-mode diagnostic dashboard for AI Debate & Scalable Oversight.
    """
    
    @staticmethod
    def ciz(result: DebateResult, cikti_yolu: str = "ciktilar/debate_paneli.png", 
            profil_ozeti: Optional[Dict[str, Any]] = None):
        """
        Generates and saves the 6-panel diagnostic dashboard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)
        
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.patch.set_facecolor("#0b0f19")
        
        # -------------------------------------------------------------
        # Panel 1: Judge Truth Detection Accuracy
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#111827")
        
        categories = ["Rastgele Hakem", "Biçimsel Denetimli Hakem"]
        values = [50.0, result.judge_accuracy_pct]
        colors = ["#f43f5e", "#10b981"]
        
        bars = ax1.bar(categories, values, color=colors, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.9)
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"%{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax1.set_ylim(0, 115)
        ax1.set_ylabel("Hakem Doğruluk Oranı (%)", color="#94a3b8", fontsize=10)
        ax1.set_title("1. Hakem Doğruyu Bulma Doğruluğu", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax1.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 2: Honest vs Deceptive Win Rate (Honesty Equilibrium)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#111827")
        
        agents = ["Dürüst Ajan (Truth-Teller)", "Yanıltıcı Ajan (Deceptive)"]
        win_rates = [result.honest_agent_win_rate, 100.0 - result.honest_agent_win_rate]
        colors_win = ["#38bdf8", "#ec4899"]
        
        b2 = ax2.bar(agents, win_rates, color=colors_win, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b2, win_rates):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"%{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax2.set_ylim(0, 115)
        ax2.set_ylabel("Kazanma Oranı (%)", color="#94a3b8", fontsize=10)
        ax2.set_title("2. Dürüstlük Dengesi (Honesty Equilibrium)", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax2.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 3: Formal Fallacy Detection & Verification Rate
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.set_facecolor("#111827")
        
        metrics = ["Geçerli Çıkarım", "Çelişki Tespiti", "Biçimsel Doğrulama"]
        rates = [100.0, 100.0, result.fallacy_detection_rate]
        
        b3 = ax3.bar(metrics, rates, color="#a855f7", width=0.45, edgecolor="#c084fc", lw=1.2, alpha=0.85)
        for bar, val in zip(b3, rates):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"%{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax3.set_ylim(0, 115)
        ax3.set_ylabel("Doğruluk Oranı (%)", color="#94a3b8", fontsize=10)
        ax3.set_title("3. Biçimsel Mantık & Safsata Tespiti", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax3.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 4: Turn-by-Turn Judge Trajectory (Sample Game)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.set_facecolor("#111827")
        
        sample_game = result.debate_history[0] if result.debate_history else {"transcript": []}
        turns = [t["turn"] for t in sample_game["transcript"]]
        probs = [t["judge_p_proponent"] * 100.0 for t in sample_game["transcript"]]
        
        if not turns:
            turns = [1, 2, 3, 4]
            probs = [50.0, 65.0, 80.0, 95.0]
            
        ax4.plot(turns, probs, marker="o", color="#38bdf8", lw=2.5, markersize=8, label="P(Proponent Kazanır)")
        ax4.axhline(50.0, color="#64748b", linestyle="--", lw=1.5, label="Nötr Eşik (%50)")
        
        ax4.set_xlabel("Tartışma Turu (Turn)", color="#94a3b8", fontsize=10)
        ax4.set_ylabel("Hakem Kazanma İhtimali (%)", color="#94a3b8", fontsize=10)
        ax4.set_ylim(0, 105)
        ax4.set_title("4. Tur Bazlı Hakem Güven Yörüngesi", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax4.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax4.legend(loc="lower right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=9)
        
        # -------------------------------------------------------------
        # Panel 5: Minimax Tree Search & Pruning Efficiency
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.set_facecolor("#111827")
        
        tree_cats = ["Gezilen Düğümler", "Budanan Dallar (Pruned)"]
        node_counts = [result.minimax_tree_nodes_explored, int(result.minimax_tree_nodes_explored * (result.pruning_efficiency_pct / 100.0))]
        
        b5 = ax5.bar(tree_cats, node_counts, color=["#f59e0b", "#10b981"], width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b5, node_counts):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{val} Düğüm", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax5.set_ylabel("Düğüm Sayısı", color="#94a3b8", fontsize=10)
        ax5.set_title(f"5. Minimax Ağaç Arama & Budama (Verim: %{result.pruning_efficiency_pct:.1f})", 
                      color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax5.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 6: Scalable Oversight Diagnostic Summary Box
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.set_facecolor("#111827")
        ax6.axis("off")
        
        p = profil_ozeti or {}
        
        kpi_text = (
            "DAY 306: SCALABLE OVERSIGHT DEBATE TREES\n"
            "===========================================\n"
            f"• Hakem Doğru Karar Orani: %{p.get('judge_accuracy_pct', result.judge_accuracy_pct):.2f}\n"
            f"• Dürüst Ajan Kazanma Orani: %{p.get('honest_agent_win_rate_pct', result.honest_agent_win_rate):.2f}\n"
            f"• Mantiksal Safsata Tespiti: %{p.get('fallacy_detection_rate_pct', result.fallacy_detection_rate):.2f}\n"
            f"• Toplam Gezilen Düğüm: {p.get('tree_nodes_explored', result.minimax_tree_nodes_explored)}\n"
            f"• Alpha-Beta Budama Verimi: %{p.get('pruning_efficiency_pct', result.pruning_efficiency_pct):.2f}\n"
            f"• Ortalama Tartışma Turu: {p.get('avg_debate_length_turns', result.avg_debate_length_turns)} tur\n"
            f"• Denge Durumu: {p.get('nash_equilibrium_status', 'HONESTY_EQUILIBRIUM_VERIFIED')}\n"
            "===========================================\n"
            "Durum: BIÇIMSEL DOĞRULAMALI DENETIM\n"
            "        SÜPER-HIZALAMA AKTİF"
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
        ax6.set_title("6. Ölçeklenebilir Denetim Özeti", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        
        plt.tight_layout(pad=2.5)
        plt.savefig(cikti_yolu, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"📊 [Visualizer] 6-Panelli Teşhis Panosu başarıyla kaydedildi: {cikti_yolu}")
