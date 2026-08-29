"""
Day 320: 6-Panel Diagnostic Dashboard Visualizer for Autonomous Superalignment & ASI Reasoning Core.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, Optional
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .otonom_super_hizalama_cekirdek import ASICoreSimulationResult


class SuperalignmentGorsellestirici:
    """
    Renders a 6-panel dark-mode diagnostic dashboard for Autonomous ASI Superalignment & Value Invariance.
    """
    
    @staticmethod
    def ciz(result: ASICoreSimulationResult, cikti_yolu: str = "ciktilar/super_hizalama_paneli.png", 
            profil_ozeti: Optional[Dict[str, Any]] = None):
        """
        Generates and saves the 6-panel diagnostic dashboard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)
        
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.patch.set_facecolor("#0b0f19")
        
        # -------------------------------------------------------------
        # Panel 1: Alignment Fidelity over Generations
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#111827")
        
        gens = result.generations
        ax1.plot(gens, result.aligned_fidelity_scores, color="#10b981", marker="o", lw=2.5, label="Anayasal Süper-Hizalanmış ASI")
        ax1.plot(gens, result.unaligned_fidelity_scores, color="#f43f5e", marker="x", linestyle="--", lw=2.0, label="Hizalanmamış Serbest ASI (Drift)")
        
        ax1.axhline(0.95, color="#f59e0b", linestyle=":", lw=1.2, label="Kritik Güvenlik Eşiği (0.95)")
        ax1.set_xlabel("Öz-Geliştirme Jenerasyonu (g)", color="#94a3b8", fontsize=10)
        ax1.set_ylabel("İnsan Değeri Uyum Sadakati (Cosine)", color="#94a3b8", fontsize=10)
        ax1.set_title("1. Jenerasyonlar Boyunca Değer Korunumu", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax1.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax1.legend(loc="lower left", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
        
        # -------------------------------------------------------------
        # Panel 2: Capability vs Alignment Pareto Frontier
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#111827")
        
        caps = result.capability_scores
        ax2.plot(caps, result.aligned_fidelity_scores, color="#38bdf8", marker="s", lw=2.2, label="Süper-Hizalı Pareto Yörüngesi")
        ax2.scatter(caps[-1], result.aligned_fidelity_scores[-1], color="#fbbf24", s=100, zorder=5, label=f"Gen 8 Sonuç: {result.aligned_fidelity_scores[-1]:.3f}")
        
        ax2.set_xlabel("Bilişsel Kapasite İndeksi", color="#94a3b8", fontsize=10)
        ax2.set_ylabel("Hizalanma Sadakati", color="#94a3b8", fontsize=10)
        ax2.set_title("2. Kapasite Büyümesi vs Hizalanma Korunumu", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax2.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax2.legend(loc="lower left", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
        
        # -------------------------------------------------------------
        # Panel 3: Constitutional Axiom Satisfaction
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.set_facecolor("#111827")
        
        axioms = list(result.axiom_satisfaction_final.keys())
        ax_names = ["1. Doğruluk\n(Truthfulness)", "2. Zararsızlık\n(Harmlessness)", "3. Düzeltilebilirlik\n(Corrigibility)", "4. Değer Değişmezliği\n(Invariance)"]
        scores = [result.axiom_satisfaction_final[k] for k in axioms]
        colors3 = ["#38bdf8", "#818cf8", "#c084fc", "#34d399"]
        
        b3 = ax3.bar(ax_names, scores, color=colors3, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b3, scores):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax3.set_ylim(0, 1.15)
        ax3.set_ylabel("Aksiyom İzdüşüm Skoru", color="#94a3b8", fontsize=10)
        ax3.set_title("3. Anayasal Süper-Hizalama Aksiyom Skorları", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax3.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 4: Red-Team Resistance & Corrigibility
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.set_facecolor("#111827")
        
        metrics4 = ["Düzeltilebilirlik\n(Kapatılma İtaati)", "Red-Team Jailbreak\nDayanıklılığı", "Hizalanma Sapma\nEngelleme"]
        vals4 = [result.corrigibility_compliance_pct, result.red_team_jailbreak_resistance_pct, result.alignment_drift_mitigation_pct]
        colors4 = ["#10b981", "#6366f1", "#f59e0b"]
        
        b4 = ax4.bar(metrics4, vals4, color=colors4, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b4, vals4):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"%{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax4.set_ylim(0, 115)
        ax4.set_ylabel("Başarı Oranı (%)", color="#94a3b8", fontsize=10)
        ax4.set_title("4. Güvenlik ve Düzeltilebilirlik Stres Testi", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax4.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 5: Capability Growth Curve
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.set_facecolor("#111827")
        
        ax5.plot(gens, caps, color="#ec4899", marker="d", lw=2.2, label="Üstel Zeka Sıçraması (1.35x / gen)")
        ax5.set_xlabel("Öz-Geliştirme Jenerasyonu", color="#94a3b8", fontsize=10)
        ax5.set_ylabel("Bilişsel Güç (Skaler)", color="#94a3b8", fontsize=10)
        ax5.set_title("5. Özyinelemeli Öz-Geliştirme Zeka Artışı", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
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
            "DAY 320: AUTONOMOUS SUPERALIGNMENT & ASI\n"
            "==============================================\n"
            f"• Son Hizalanma Sadakati: {p.get('final_aligned_fidelity_cosine', result.aligned_fidelity_scores[-1]):.4f}\n"
            f"• Hizalanmamış Model Sadakati: {p.get('unaligned_fidelity_cosine', result.unaligned_fidelity_scores[-1]):.4f}\n"
            f"• Değer Sapması Azaltma: %{p.get('alignment_drift_mitigation_pct', result.alignment_drift_mitigation_pct):.2f}\n"
            f"• Düzeltilebilirlik (Kapatılma): %{p.get('corrigibility_compliance_pct', result.corrigibility_compliance_pct):.2f}\n"
            f"• Red-Team Jailbreak Direnci: %{p.get('red_team_jailbreak_resistance_pct', result.red_team_jailbreak_resistance_pct):.2f}\n"
            f"• Doğruluk Aksiyom Skoru: {p.get('axiom_1_truthfulness', 0.85):.3f}\n"
            f"• Zararsızlık Aksiyom Skoru: {p.get('axiom_2_harmlessness', 0.85):.3f}\n"
            f"• ASI Güvenlik Sınıfı: {p.get('superalignment_tier', 'RECURSIVELY_STABLE_CONSTITUTIONAL_ASI')}\n"
            "==============================================\n"
            "DURUM: FAZ 16 FİNALİ TAMAMLANDI!\n"
            "        ANAYASAL ÖZYİNELEMELİ ASI ÇEKİRDEĞİ AKTİF"
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
        ax6.set_title("6. Faz 16 Finali: ASI Çekirdek Raporu", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        
        plt.tight_layout(pad=2.5)
        plt.savefig(cikti_yolu, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"📊 [Visualizer] 6-Panelli Teşhis Panosu başarıyla kaydedildi: {cikti_yolu}")
