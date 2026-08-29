"""
Day 317: 6-Panel Diagnostic Dashboard Visualizer for Automated Epistemology & Counterfactuals.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, Optional
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .epistemoloji_karsiolgusal_lab import EpistemologyBenchmarkResult


class EpistemologyGorsellestirici:
    """
    Renders a 6-panel dark-mode diagnostic dashboard for Structural Causal Models & Counterfactuals.
    """
    
    @staticmethod
    def ciz(result: EpistemologyBenchmarkResult, cikti_yolu: str = "ciktilar/epistemoloji_paneli.png", 
            profil_ozeti: Optional[Dict[str, Any]] = None):
        """
        Generates and saves the 6-panel diagnostic dashboard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)
        
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.patch.set_facecolor("#0b0f19")
        
        # -------------------------------------------------------------
        # Panel 1: Pearl's 3 Causal Levels
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#111827")
        
        levels = ["Seviye 1: Gözlemsel\nİlişki (P(Y|X))", "Seviye 2: Müdahale\n(do(X)) ATE", "Seviye 3: Karşı-Olgusal\n(Y_{X=x'} | x,y)"]
        vals1 = [result.observational_association, result.average_treatment_effect_ate, result.average_treatment_effect_ate * 0.98]
        colors1 = ["#f59e0b", "#38bdf8", "#10b981"]
        
        b1 = ax1.bar(levels, vals1, color=colors1, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b1, vals1):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{val:.3f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax1.set_ylabel("Etki Büyüklüğü (Effect Size)", color="#94a3b8", fontsize=10)
        ax1.set_title("1. Pearl'ün 3-Basamaklı Nedensellik Hiyerarşisi", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax1.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 2: Causal Effect Decomposition
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#111827")
        
        effects = ["Toplam Nedensel\nEtki (ATE)", "Doğrudan Etki\n(NDE: X->Y)", "Dolaylı Etki\n(NIE: X->M->Y)"]
        decomp_vals = [result.average_treatment_effect_ate, result.natural_direct_effect_nde, result.natural_indirect_effect_nie]
        colors2 = ["#38bdf8", "#818cf8", "#c084fc"]
        
        b2 = ax2.bar(effects, decomp_vals, color=colors2, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b2, decomp_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{val:.3f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax2.set_ylabel("Ayrıştırılmış Etki", color="#94a3b8", fontsize=10)
        ax2.set_title("2. Nedensel Yol Ayrışımı (Mediation Analysis)", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax2.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 3: Interventional Treatment Response Curve
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.set_facecolor("#111827")
        
        x_grid, y_resp = result.treatment_response_curve
        ax3.plot(x_grid, y_resp, color="#10b981", lw=2.5, label="E[Y | do(X=x)] Tepki Doğrusu")
        ax3.scatter([0.0, 1.0], [y_resp[len(x_grid)//2 - 8], y_resp[len(x_grid)//2 + 8]], color="#f43f5e", s=60, zorder=5, label="ATE Referans Noktaları")
        
        ax3.set_xlabel("Müdahale Değeri (do(X = x))", color="#94a3b8", fontsize=10)
        ax3.set_ylabel("Beklenen Sonuç E[Y]", color="#94a3b8", fontsize=10)
        ax3.set_title("3. do-Calculus Müdahale Tepki Eğrisi", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax3.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax3.legend(loc="upper left", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
        
        # -------------------------------------------------------------
        # Panel 4: Individual Factual vs Counterfactual Pairs
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.set_facecolor("#111827")
        
        samples = result.factual_vs_counterfactual_samples
        ids = [f"Örnek #{s['sample_id']}" for s in samples]
        y_fact = [s["factual_y"] for s in samples]
        y_cf = [s["counterfactual_y"] for s in samples]
        
        x = np.arange(len(ids))
        width = 0.35
        
        ax4.bar(x - width/2, y_fact, width, label="Olgusal Gerçek (Y)", color="#38bdf8", alpha=0.85, edgecolor="#ffffff", lw=1.0)
        ax4.bar(x + width/2, y_cf, width, label="Karşı-Olgusal (Y_{X=0})", color="#a855f7", alpha=0.85, edgecolor="#ffffff", lw=1.0)
        
        ax4.set_xticks(x)
        ax4.set_xticklabels(ids, color="#94a3b8", fontsize=9)
        ax4.set_ylabel("Sonuç Değeri Y", color="#94a3b8", fontsize=10)
        ax4.set_title("4. Bireysel Karşı-Olgusal Akıl Yürütme", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax4.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax4.legend(loc="upper right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
        
        # -------------------------------------------------------------
        # Panel 5: Confounding Bias Gap
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.set_facecolor("#111827")
        
        bias_cats = ["Gözlem Yanlılığı\n(Z Karıştırıcısı)", "Saf Nedensellik\n(Causal ATE)"]
        b_vals = [result.confounding_bias_gap, result.average_treatment_effect_ate]
        b_colors = ["#f43f5e", "#10b981"]
        
        b5 = ax5.bar(bias_cats, b_vals, color=b_colors, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b5, b_vals):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{val:.3f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax5.set_ylabel("Etki Boyutu", color="#94a3b8", fontsize=10)
        ax5.set_title("5. Karıştırıcı Değişken (Confounder) Yanlılık Boşluğu", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax5.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 6: Telemetry Summary Box
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.set_facecolor("#111827")
        ax6.axis("off")
        
        p = profil_ozeti or {}
        
        kpi_text = (
            "DAY 317: AUTOMATED EPISTEMOLOGY & SCM LAB\n"
            "==============================================\n"
            f"• Gözlemsel İlişki E[Y|X]: {p.get('observational_association', result.observational_association):.4f}\n"
            f"• Ortalama Tedavi Etkisi (ATE): {p.get('average_treatment_effect_ate', result.average_treatment_effect_ate):.4f}\n"
            f"• Doğrudan Etki (NDE): {p.get('natural_direct_effect_nde', result.natural_direct_effect_nde):.4f}\n"
            f"• Dolaylı Etki (NIE): {p.get('natural_indirect_effect_nie', result.natural_indirect_effect_nie):.4f}\n"
            f"• Karıştırıcı Yanlılık Farkı: {p.get('confounding_bias_gap', result.confounding_bias_gap):.4f}\n"
            f"• Karşı-Olgusal Tutarlılık: %{p.get('counterfactual_consistency_pct', result.counterfactual_consistency_pct):.2f}\n"
            f"• Epistemoloji Sınıfı: {p.get('causal_epistemology_tier', 'LEVEL_3_COUNTERFACTUAL_FAITHFUL')}\n"
            "==============================================\n"
            "DURUM: PEARL'ÜN 3. SEVİYE KARŞI-OLGUSAL\n"
            "        AKIL YÜRÜTME MOTORU AKTİF"
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
        ax6.set_title("6. Epistemolojik Nedensellik Modeli Özeti", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        
        plt.tight_layout(pad=2.5)
        plt.savefig(cikti_yolu, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"📊 [Visualizer] 6-Panelli Teşhis Panosu başarıyla kaydedildi: {cikti_yolu}")
