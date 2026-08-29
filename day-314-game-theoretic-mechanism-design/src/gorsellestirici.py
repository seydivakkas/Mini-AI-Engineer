"""
Day 314: 6-Panel Diagnostic Dashboard Visualizer for Game-Theoretic Mechanism Design & Nash Bargaining.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, Optional
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .oyun_teorisi_mekanizma import MechanismResult


class GameTheoreticGorsellestirici:
    """
    Renders a 6-panel dark-mode diagnostic dashboard for VCG Mechanism & Nash Bargaining.
    """
    
    @staticmethod
    def ciz(result: MechanismResult, cikti_yolu: str = "ciktilar/oyun_teorisi_paneli.png", 
            profil_ozeti: Optional[Dict[str, Any]] = None):
        """
        Generates and saves the 6-panel diagnostic dashboard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)
        
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.patch.set_facecolor("#0b0f19")
        
        agents = list(result.vcg_payments.keys())
        
        # -------------------------------------------------------------
        # Panel 1: VCG Net Utilities
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#111827")
        
        utils = [result.vcg_net_utilities[a] for a in agents]
        b1 = ax1.bar(agents, utils, color="#10b981", width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b1, utils):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=9, fontweight="bold")
                     
        ax1.set_ylabel("Net Fayda (Utility)", color="#94a3b8", fontsize=10)
        ax1.set_title("1. VCG Net Ajan Faydaları (u_i = v_i - p_i)", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax1.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 2: VCG Payments (Externalities)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#111827")
        
        pays = [result.vcg_payments[a] for a in agents]
        b2 = ax2.bar(agents, pays, color="#f59e0b", width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b2, pays):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f"{val:.2f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=9, fontweight="bold")
                     
        ax2.set_ylabel("Dışsallık Ödemesi (p_i)", color="#94a3b8", fontsize=10)
        ax2.set_title("2. VCG Teşvik Uyumlu Dışsallık Ödemeleri", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax2.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 3: DSIC Incentive Compatibility Verification
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.set_facecolor("#111827")
        
        strat = ["Dürüst Teklif (Truthful)", "Stratejik Manipülasyon (Lying)"]
        alpha_u = result.vcg_net_utilities[agents[0]]
        lying_u = alpha_u - result.truthful_vs_manipulated_utility_gain
        strat_vals = [alpha_u, lying_u]
        colors3 = ["#10b981", "#f43f5e"]
        
        b3 = ax3.bar(strat, strat_vals, color=colors3, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b3, strat_vals):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax3.set_ylabel("Ajan-Alpha Kazancı", color="#94a3b8", fontsize=10)
        ax3.set_title("3. DSIC Baskın Strateji Doğrulaması", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax3.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 4: Nash Bargaining Resource Distribution
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.set_facecolor("#111827")
        
        alloc_vals = [result.nash_bargaining_allocations[a] for a in agents]
        colors_pie = ["#38bdf8", "#818cf8", "#c084fc", "#f472b6"]
        
        ax4.pie(alloc_vals, labels=agents, autopct="%1.1f%%", colors=colors_pie, startangle=140,
                textprops=dict(color="#f8fafc", fontsize=9, fontweight="bold"),
                wedgeprops=dict(width=0.45, edgecolor="#0b0f19", lw=2))
        ax4.set_title("4. Nash Pazarlığı Hesaplama Kaynağı Dağılımı", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        
        # -------------------------------------------------------------
        # Panel 5: Individual Agent Surpluses above Threat Point
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.set_facecolor("#111827")
        
        surplus_vals = [result.nash_net_surpluses[a] for a in agents]
        b5 = ax5.bar(agents, surplus_vals, color="#a855f7", width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b5, surplus_vals):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"+{val:.2f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=9, fontweight="bold")
                     
        ax5.set_ylabel("Net Rant Fazlası (u_i - d_i)", color="#94a3b8", fontsize=10)
        ax5.set_title("5. Tehdit Noktası Üzerindeki Bireysel Artıklar", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax5.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 6: Telemetry Summary Box
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.set_facecolor("#111827")
        ax6.axis("off")
        
        p = profil_ozeti or {}
        
        kpi_text = (
            "DAY 314: GAME THEORETIC MECHANISM DESIGN\n"
            "==============================================\n"
            f"• VCG Optimal Çıktı: #{p.get('vcg_optimal_outcome', result.vcg_optimal_outcome)}\n"
            f"• Toplam Sosyal Refah: {p.get('vcg_social_welfare', result.vcg_social_welfare):.2f}\n"
            f"• Toplam VCG Ödemeleri: {p.get('vcg_total_payments', sum(result.vcg_payments.values())):.2f}\n"
            f"• DSIC Dürüstlük Kazanç Farkı: +{p.get('dsic_truthful_stability_gain', result.truthful_vs_manipulated_utility_gain):.4f}\n"
            f"• Nash Pazarlık Çarpımı: {p.get('nash_bargaining_product', result.total_nash_product):.4f}\n"
            f"• Pareto Etkinliği: %{p.get('pareto_efficiency_pct', result.pareto_efficiency_pct):.2f}\n"
            f"• Denge Sınıfı: {p.get('equilibrium_tier', 'PARETO_OPTIMAL_DSIC_STABLE')}\n"
            "==============================================\n"
            "DURUM: VCG TEŞVİK UYUMLULUĞU VE GENELLEŞTİRİLMİŞ\n"
            "        NASH PAZARLIK ÇÖZÜMÜ AKTİF"
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
        ax6.set_title("6. Oyun Teorisi & Mekanizma Özeti", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        
        plt.tight_layout(pad=2.5)
        plt.savefig(cikti_yolu, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"📊 [Visualizer] 6-Panelli Teşhis Panosu başarıyla kaydedildi: {cikti_yolu}")
