"""
Day 318: 6-Panel Diagnostic Dashboard Visualizer for Neuro-Symbolic Continuous Logic.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, Optional
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .noro_sembolik_mantik import NeuroSymbolicResult, ContinuousLogicEngine, TNormType


class NeuroSymbolicGorsellestirici:
    """
    Renders a 6-panel dark-mode diagnostic dashboard for Neuro-Symbolic Continuous Logic & Theorem Prover.
    """
    
    @staticmethod
    def ciz(result: NeuroSymbolicResult, cikti_yolu: str = "ciktilar/noro_sembolik_paneli.png", 
            profil_ozeti: Optional[Dict[str, Any]] = None):
        """
        Generates and saves the 6-panel diagnostic dashboard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)
        
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.patch.set_facecolor("#0b0f19")
        
        # -------------------------------------------------------------
        # Panel 1: Training & Logic Loss Convergence
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#111827")
        
        steps = np.arange(1, len(result.loss_history) + 1)
        ax1.plot(steps, result.loss_history, color="#38bdf8", lw=2.2, label="Toplam Nöro-Sembolik Kayıp")
        
        ax1.set_xlabel("Eğitim Adımı (Step)", color="#94a3b8", fontsize=10)
        ax1.set_ylabel("Kayıp (Loss)", color="#94a3b8", fontsize=10)
        ax1.set_title("1. Nöro-Sembolik Gradyan Optimizasyonu", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax1.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
        
        # -------------------------------------------------------------
        # Panel 2: Axiom Satisfaction Evolution
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#111827")
        
        ax2.plot(steps, result.rule_satisfaction_history, color="#10b981", lw=2.2, label="Ortalama Aksiyom Sağlanma Oranı (%)")
        ax2.axhline(100.0, color="#64748b", linestyle="--", alpha=0.5)
        
        ax2.set_xlabel("Eğitim Adımı (Step)", color="#94a3b8", fontsize=10)
        ax2.set_ylabel("Aksiyom Doğrulanma (%)", color="#94a3b8", fontsize=10)
        ax2.set_title("2. Mantıksal Kurallara Uyum Gelişimi", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax2.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax2.legend(loc="lower right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
        
        # -------------------------------------------------------------
        # Panel 3: Individual Axiom Satisfaction Rates
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.set_facecolor("#111827")
        
        axioms = list(result.rule_satisfaction_rates.keys())
        short_names = ["Aksiyom 1\n(Taban Kural)", "Aksiyom 2\n(Geçişlilik)", "Aksiyom 3\n(Asimetri)"]
        rates = [result.rule_satisfaction_rates[k] * 100.0 for k in axioms]
        colors3 = ["#818cf8", "#38bdf8", "#34d399"]
        
        b3 = ax3.bar(short_names, rates, color=colors3, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b3, rates):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"%{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax3.set_ylim(0, 115)
        ax3.set_ylabel("Kural Doğruluk Skoru (%)", color="#94a3b8", fontsize=10)
        ax3.set_title("3. Birinci Dereceden Mantık (FOL) Aksiyom Skorları", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax3.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 4: Proven Test Queries
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.set_facecolor("#111827")
        
        queries = [q["query"] for q in result.proven_queries]
        t_vals = [q["truth_value"] for q in result.proven_queries]
        q_colors = ["#10b981" if q["is_proven"] else "#f43f5e" for q in result.proven_queries]
        
        b4 = ax4.barh(queries, t_vals, color=q_colors, height=0.45, edgecolor="#ffffff", lw=1.0, alpha=0.85)
        for bar, val in zip(b4, t_vals):
            ax4.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", 
                     ha="left", va="center", color="#f8fafc", fontsize=9.5, fontweight="bold")
                     
        ax4.axvline(0.50, color="#f59e0b", linestyle="--", lw=1.2, label="Karar Eşiği (0.50)")
        ax4.set_xlim(0, 1.25)
        ax4.set_xlabel("Hesaplanan Sürekli Doğruluk Değeri tau(phi)", color="#94a3b8", fontsize=10)
        ax4.set_title("4. Türetilebilir Teorem Kanıtlama Başarısı", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax4.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax4.legend(loc="lower right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
        
        # -------------------------------------------------------------
        # Panel 5: Fuzzy T-Norm Comparison
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.set_facecolor("#111827")
        
        a_vals = np.linspace(0.0, 1.0, 100)
        fixed_b = 0.70
        b_arr = np.full_like(a_vals, fixed_b)
        
        prod_t = ContinuousLogicEngine.conjunction(a_vals, b_arr, TNormType.PRODUCT)
        luka_t = ContinuousLogicEngine.conjunction(a_vals, b_arr, TNormType.LUKASIEWICZ)
        godel_t = ContinuousLogicEngine.conjunction(a_vals, b_arr, TNormType.GODEL)
        
        ax5.plot(a_vals, prod_t, color="#f59e0b", lw=2.0, label="Product: a * 0.7")
        ax5.plot(a_vals, luka_t, color="#38bdf8", lw=2.0, label="Łukasiewicz: max(0, a+0.7-1)")
        ax5.plot(a_vals, godel_t, color="#a855f7", lw=2.0, label="Gödel: min(a, 0.7)")
        
        ax5.set_xlabel("Doğruluk Değeri a (b = 0.70)", color="#94a3b8", fontsize=10)
        ax5.set_ylabel("T-Norm Birleşim T(a, 0.7)", color="#94a3b8", fontsize=10)
        ax5.set_title("5. T-Norm Sürekli Mantık Bağlaçları Kıyası", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax5.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax5.legend(loc="upper left", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
        
        # -------------------------------------------------------------
        # Panel 6: Telemetry Summary Box
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.set_facecolor("#111827")
        ax6.axis("off")
        
        p = profil_ozeti or {}
        
        kpi_text = (
            "DAY 318: NEURO-SYMBOLIC CONTINUOUS LOGIC\n"
            "==============================================\n"
            f"• T-Norm Çerçevesi: {p.get('t_norm_framework', result.t_norm_name.upper())}\n"
            f"• Teorem Kanıtlama Başarısı: %{p.get('theorem_proof_accuracy_pct', result.theorem_proof_accuracy_pct):.2f}\n"
            f"• Ortalama Kural Uyum Oranı: %{p.get('mean_axiom_satisfaction_pct', 85.0):.2f}\n"
            f"• Taban Kuralı Sağlanma: %{p.get('axiom_1_base_sat', 0.0)*100:.1f}\n"
            f"• Geçişlilik Kuralı Sağlanma: %{p.get('axiom_2_transitivity_sat', 0.0)*100:.1f}\n"
            f"• Asimetri Kuralı Sağlanma: %{p.get('axiom_3_asymmetry_sat', 0.0)*100:.1f}\n"
            f"• Son Toplam Kayıp: {p.get('final_total_loss', result.total_loss):.4f}\n"
            f"• Mantıksal İhlal Kaybı: {p.get('logical_violation_loss', result.final_logical_violation_loss):.4f}\n"
            f"• Nöro-Sembolik Sınıfı: {p.get('neuro_symbolic_tier', 'LOGIC_GROUNDED_NEURO_SYMBOLIC')}\n"
            "==============================================\n"
            "DURUM: TÜRETEBİLİR BİRİNCİ DERECEDEN\n"
            "        BULANIK TEOREM KANITLAYICI AKTİF"
        )
        
        ax6.text(
            0.05, 0.5, kpi_text,
            transform=ax6.transAxes,
            fontsize=8.8,
            fontfamily="monospace",
            color="#e2e8f0",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#1e293b", edgecolor="#38bdf8", lw=1.5, alpha=0.95)
        )
        ax6.set_title("6. Nöro-Sembolik Çıkarım Motoru Özeti", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        
        plt.tight_layout(pad=2.5)
        plt.savefig(cikti_yolu, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"📊 [Visualizer] 6-Panelli Teşhis Panosu başarıyla kaydedildi: {cikti_yolu}")
