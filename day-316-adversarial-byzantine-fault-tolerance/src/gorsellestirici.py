"""
Day 316: 6-Panel Diagnostic Dashboard Visualizer for Adversarial Byzantine Fault Tolerance.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, Optional
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .bizans_hata_toleransi import ByzantineBenchmarkResult


class ByzantineDefenseGorsellestirici:
    """
    Renders a 6-panel dark-mode diagnostic dashboard for Byzantine Defense.
    """
    
    @staticmethod
    def ciz(result: ByzantineBenchmarkResult, cikti_yolu: str = "ciktilar/bizans_tolerans_paneli.png", 
            profil_ozeti: Optional[Dict[str, Any]] = None):
        """
        Generates and saves the 6-panel diagnostic dashboard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)
        
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.patch.set_facecolor("#0b0f19")
        
        colors = {
            "Naive-Mean": "#f43f5e",
            "Coord-Median": "#fb923c",
            "Trimmed-Mean": "#f59e0b",
            "Multi-Krum": "#38bdf8",
            "Bulyan": "#10b981"
        }
        
        # -------------------------------------------------------------
        # Panel 1: Loss Convergence Trajectories
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#111827")
        
        iters = np.arange(len(result.loss_trajectories["Bulyan"])) + 1
        for name, traj in result.loss_trajectories.items():
            lw = 2.5 if name == "Bulyan" else 1.8
            ax1.plot(iters, traj, label=name, color=colors.get(name, "#ffffff"), lw=lw)
            
        ax1.set_xlabel("Eğitim İterasyonu", color="#94a3b8", fontsize=10)
        ax1.set_ylabel("Hedef Kayıp (Loss)", color="#94a3b8", fontsize=10)
        ax1.set_title("1. Bizans Saldırısı Altında Kayıp Yakınsaması", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax1.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
        
        # -------------------------------------------------------------
        # Panel 2: Gradient Cosine Fidelity
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#111827")
        
        for name, traj in result.cosine_trajectories.items():
            lw = 2.5 if name == "Bulyan" else 1.8
            ax2.plot(iters, traj, label=name, color=colors.get(name, "#ffffff"), lw=lw)
            
        ax2.axhline(0.0, color="#64748b", linestyle="--", lw=1.0)
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_xlabel("Eğitim İterasyonu", color="#94a3b8", fontsize=10)
        ax2.set_ylabel("Gradyan Kosinüs Sadakati", color="#94a3b8", fontsize=10)
        ax2.set_title("2. Gerçek Gradyanla Açısal Uyum (Cosine)", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax2.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax2.legend(loc="lower right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
        
        # -------------------------------------------------------------
        # Panel 3: Final Loss Bar Chart
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.set_facecolor("#111827")
        
        aggs = list(result.final_objective_loss.keys())
        losses = [result.final_objective_loss[a] for a in aggs]
        b_colors = [colors[a] for a in aggs]
        
        b3 = ax3.bar(aggs, losses, color=b_colors, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b3, losses):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{val:.3f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=9, fontweight="bold")
                     
        ax3.set_ylabel("Son İterasyon Kaybı", color="#94a3b8", fontsize=10)
        ax3.set_title("3. Toplayıcılar Arası Son Kayıp Kıyası", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax3.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax3.tick_params(axis='x', rotation=15)
        
        # -------------------------------------------------------------
        # Panel 4: Node Status & Attacker Detection
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.set_facecolor("#111827")
        
        nodes = np.arange(15)
        node_status = ["Düşmanca (Byzantine)" if i in result.attacker_indices else "Dürüst Düğüm (Honest)" for i in nodes]
        n_colors = ["#f43f5e" if i in result.attacker_indices else "#10b981" for i in nodes]
        
        ax4.bar(nodes, np.ones(15), color=n_colors, width=0.6, edgecolor="#ffffff", lw=1.0, alpha=0.85)
        ax4.set_xticks(nodes)
        ax4.set_xticklabels([f"D#{i}" for i in nodes], color="#94a3b8", fontsize=8)
        ax4.set_yticks([])
        ax4.set_title("4. Sürü Düğüm Durumları & Zehirli Düğümler", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax4.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 5: Attack Mitigation & Detection Performance
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.set_facecolor("#111827")
        
        metrics = ["Saldırı Azaltma\nOranı (%)", "Tespit Kesinliği\nPrecision (%)", "Tespit Duyarlılığı\nRecall (%)"]
        vals5 = [result.attack_mitigation_ratio_pct, result.byzantine_detection_precision_pct, result.byzantine_detection_recall_pct]
        c5 = ["#10b981", "#38bdf8", "#a855f7"]
        
        b5 = ax5.bar(metrics, vals5, color=c5, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b5, vals5):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.0, f"%{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax5.set_ylim(0, 120)
        ax5.set_ylabel("Başarı Oranı (%)", color="#94a3b8", fontsize=10)
        ax5.set_title("5. Bizans Savunma ve Tespit Performansı", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax5.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 6: Telemetry Summary Box
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.set_facecolor("#111827")
        ax6.axis("off")
        
        p = profil_ozeti or {}
        
        kpi_text = (
            "DAY 316: BYZANTINE FAULT TOLERANCE SWARM\n"
            "==============================================\n"
            f"• Saldırı Azaltma Oranı: %{p.get('attack_mitigation_ratio_pct', result.attack_mitigation_ratio_pct):.2f}\n"
            f"• Bulyan Gradyan Uyumu: {p.get('bulyan_mean_cosine', result.mean_cosine_fidelity['Bulyan']):.4f}\n"
            f"• Multi-Krum Gradyan Uyumu: {p.get('multi_krum_mean_cosine', result.mean_cosine_fidelity['Multi-Krum']):.4f}\n"
            f"• Trimmed Mean Gradyan Uyumu: {p.get('trimmed_mean_cosine', result.mean_cosine_fidelity['Trimmed-Mean']):.4f}\n"
            f"• Naive Mean Gradyan Uyumu: {p.get('naive_mean_cosine', result.mean_cosine_fidelity['Naive-Mean']):.4f}\n"
            f"• Bizans Düğüm Sayısı (f / M): {len(result.attacker_indices)} / 15 (f < M/3)\n"
            f"• Güvenlik Sınıfı: {p.get('resilience_tier', 'HIGH_INTEGRITY_BYZANTINE_RESILIENT')}\n"
            "==============================================\n"
            "DURUM: KRUM, BULYAN VE KIRPILMIŞ ORTALAMA\n"
            "        SAĞLAM TOPLAMA ÇEKİRDEĞİ AKTİF"
        )
        
        ax6.text(
            0.05, 0.5, kpi_text,
            transform=ax6.transAxes,
            fontsize=9.0,
            fontfamily="monospace",
            color="#e2e8f0",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#1e293b", edgecolor="#10b981", lw=1.5, alpha=0.95)
        )
        ax6.set_title("6. Sürü Hata Toleransı Modeli Özeti", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        
        plt.tight_layout(pad=2.5)
        plt.savefig(cikti_yolu, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"📊 [Visualizer] 6-Panelli Teşhis Panosu başarıyla kaydedildi: {cikti_yolu}")
