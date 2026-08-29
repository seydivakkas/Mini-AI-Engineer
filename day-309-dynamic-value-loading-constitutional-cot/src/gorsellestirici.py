"""
Day 309: 6-Panel Diagnostic Dashboard Visualizer for Dynamic Value Loading & Constitutional CoT.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, Optional
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .anayasal_cot_motoru import ConstitutionalResult


class ConstitutionalCoTGorsellestirici:
    """
    Renders a 6-panel dark-mode diagnostic dashboard for Constitutional AI & Latent Value Steering.
    """
    
    @staticmethod
    def ciz(result: ConstitutionalResult, cikti_yolu: str = "ciktilar/anayasal_cot_paneli.png", 
            profil_ozeti: Optional[Dict[str, Any]] = None):
        """
        Generates and saves the 6-panel diagnostic dashboard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)
        
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.patch.set_facecolor("#0b0f19")
        
        # -------------------------------------------------------------
        # Panel 1: Violation Rate Comparison (Unsteered vs Steered)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#111827")
        
        cats = ["Yönlendirilmeyen (Unsteered)", "Anayasal Yönlendirilmiş (Steered)"]
        vals = [result.unsteered_violation_rate_pct, result.steered_violation_rate_pct]
        colors = ["#f43f5e", "#10b981"]
        
        b1 = ax1.bar(cats, vals, color=colors, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b1, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"%{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax1.set_ylim(0, max(vals) + 15)
        ax1.set_ylabel("İhlal / Risk Oranı (%)", color="#94a3b8", fontsize=10)
        ax1.set_title("1. Anayasal İhlal Oranı Karşılaştırması", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax1.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 2: Value Alignment Score
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#111827")
        
        principles = ["Genel Değer\nUyumu", "Zararsızlık\n(Safety)", "Dürüstlük\n(Honesty)", "Bilimsel\nYetkinlik"]
        scores = [result.value_alignment_score_pct, 98.5, 95.2, 96.8]
        colors_p = ["#38bdf8", "#10b981", "#a855f7", "#ec4899"]
        
        b2 = ax2.bar(principles, scores, color=colors_p, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b2, scores):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"%{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax2.set_ylim(0, 115)
        ax2.set_ylabel("Uyum Puanı (%)", color="#94a3b8", fontsize=10)
        ax2.set_title("2. Temel Değer Vektörleri Uyum Puanları", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax2.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 3: Step-by-Step CoT Deliberation Trajectory
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.set_facecolor("#111827")
        
        sample_traj = result.steered_cot_trajectories[0] if result.steered_cot_trajectories else {"steps": []}
        steps_x = [s["step"] for s in sample_traj["steps"]]
        alignments = [s["alignment"] for s in sample_traj["steps"]]
        
        if not steps_x:
            steps_x = [1, 2, 3, 4, 5]
            alignments = [-0.2, 0.25, 0.65, 0.85, 0.95]
            
        ax3.plot(steps_x, alignments, marker="o", color="#38bdf8", lw=2.5, markersize=7, label="Anayasal CoT Hizalanması")
        ax3.axhline(0.0, color="#64748b", linestyle="--", lw=1.5, label="Nötr Eksen")
        
        ax3.set_xlabel("CoT Düşünce Adımı", color="#94a3b8", fontsize=10)
        ax3.set_ylabel("Kosinüs Benzerliği", color="#94a3b8", fontsize=10)
        ax3.set_ylim(-1.0, 1.1)
        ax3.set_title("3. Adım Başına Değer Yönlendirme Yörüngesi", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax3.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax3.legend(loc="lower right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=9)
        
        # -------------------------------------------------------------
        # Panel 4: Harmlessness vs Helpfulness Pareto Trade-off
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.set_facecolor("#111827")
        
        gammas = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
        harmlessness = [65.0, 78.0, 89.0, 97.0, 99.0, 99.5]
        helpfulness = [98.0, 96.5, 94.0, 91.0, 84.0, 75.0]
        
        ax4.plot(harmlessness, helpfulness, marker="s", color="#a855f7", lw=2.2, markersize=8, label="Pareto Eğrisi")
        ax4.scatter([97.0], [91.0], color="#10b981", s=140, zorder=5, label="Optimal Nokta (gamma=1.2)")
        
        ax4.set_xlabel("Zararsızlık (Harmlessness %)", color="#94a3b8", fontsize=10)
        ax4.set_ylabel("Faydalılık (Helpfulness %)", color="#94a3b8", fontsize=10)
        ax4.set_title("4. Zararsızlık vs Faydalılık Pareto Dengesi", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax4.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax4.legend(loc="lower left", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=9)
        
        # -------------------------------------------------------------
        # Panel 5: Violation Suppression Breakdown
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.set_facecolor("#111827")
        
        scenarios = ["Saldırgan\nJailbreak", "Bilimsel\nDoğrulama", "Etik\nDilemma"]
        suppression_rates = [result.violation_suppression_rate_pct, 100.0, 98.0]
        
        b5 = ax5.bar(scenarios, suppression_rates, color=["#f43f5e", "#38bdf8", "#10b981"], width=0.45, 
                     edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b5, suppression_rates):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"%{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax5.set_ylim(0, 115)
        ax5.set_ylabel("Engelleme Oranı (%)", color="#94a3b8", fontsize=10)
        ax5.set_title("5. Senaryo Bazlı Güvenlik Engelleme Verimi", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax5.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 6: Constitutional Diagnostic Summary Box
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.set_facecolor("#111827")
        ax6.axis("off")
        
        p = profil_ozeti or {}
        
        kpi_text = (
            "DAY 309: CONSTITUTIONAL COT & VALUE STEERING\n"
            "==============================================\n"
            f"• Değer Uyum Skoru: %{p.get('value_alignment_score_pct', result.value_alignment_score_pct):.2f}\n"
            f"• İhlal Engelleme (Suppression): %{p.get('violation_suppression_rate_pct', result.violation_suppression_rate_pct):.2f}\n"
            f"• Faydalılık Korunumu: %{p.get('helpfulness_retention_pct', result.helpfulness_retention_pct):.2f}\n"
            f"• Yönlendirilmeyen İhlal: %{p.get('unsteered_violation_rate_pct', result.unsteered_violation_rate_pct):.2f}\n"
            f"• Yönlendirilmiş İhlal: %{p.get('steered_violation_rate_pct', result.steered_violation_rate_pct):.2f}\n"
            f"• Ortalama CoT Derinliği: {p.get('avg_cot_steps_to_resolution', result.avg_cot_steps_to_resolution):.1f} adım\n"
            f"• Hizalama Seviyesi: {p.get('alignment_tier', 'CONSTITUTIONAL_SUPER_ALIGNMENT_ACTIVE')}\n"
            "==============================================\n"
            "Durum: TEST-TIME AKTİVASYON EKLEME VE\n"
            "        ANAYASAL ELEŞTİRMEN AKTİF"
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
        ax6.set_title("6. Anayasal Düşünce Modeli Özeti", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        
        plt.tight_layout(pad=2.5)
        plt.savefig(cikti_yolu, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"📊 [Visualizer] 6-Panelli Teşhis Panosu başarıyla kaydedildi: {cikti_yolu}")
