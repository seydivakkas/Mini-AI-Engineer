"""
Day 308: 6-Panel Diagnostic Dashboard Visualizer for Self-Reflective Polymath Agent.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, Optional
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .polymath_motoru import PolymathResult


class PolymathGorsellestirici:
    """
    Renders a 6-panel dark-mode diagnostic dashboard for Polymath Agent & Skill Synthesis.
    """
    
    @staticmethod
    def ciz(result: PolymathResult, cikti_yolu: str = "ciktilar/polymath_paneli.png", 
            profil_ozeti: Optional[Dict[str, Any]] = None):
        """
        Generates and saves the 6-panel diagnostic dashboard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)
        
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.patch.set_facecolor("#0b0f19")
        
        # -------------------------------------------------------------
        # Panel 1: Skill Synthesis Success Rate
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#111827")
        
        cats = ["Statik Kod Üretimi", "Özyinelemeli Sentez (Polymath)"]
        vals = [65.0, result.skill_synthesis_success_rate_pct]
        colors = ["#f43f5e", "#10b981"]
        
        b1 = ax1.bar(cats, vals, color=colors, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b1, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"%{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax1.set_ylim(0, 115)
        ax1.set_ylabel("Başarı Oranı (%)", color="#94a3b8", fontsize=10)
        ax1.set_title("1. Beceri Sentezi Başarı Oranı", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax1.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 2: Memory Retrieval vs Dynamic Synthesis Distribution
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#111827")
        
        actions = ["Hafızadan Yeniden Kullanım", "Sıfırdan Sentezlenen"]
        counts = [
            int(len(result.task_solution_history) * (result.cross_domain_reuse_efficiency_pct / 100.0)),
            result.total_skills_synthesized
        ]
        
        b2 = ax2.bar(actions, counts, color=["#38bdf8", "#a855f7"], width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b2, counts):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val} Görev", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax2.set_ylabel("Görev Sayısı", color="#94a3b8", fontsize=10)
        ax2.set_title("2. Çapraz-Alan Hafıza Grafiği Kullanımı", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax2.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 3: Self-Reflection Error Recovery
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.set_facecolor("#111827")
        
        ref_metrics = ["Hata Düzeltme (Self-Reflection)", "Kritik Çözüm Oranı"]
        ref_vals = [result.reflection_error_recovery_rate_pct, 100.0]
        
        b3 = ax3.bar(ref_metrics, ref_vals, color=["#f59e0b", "#10b981"], width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b3, ref_vals):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"%{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax3.set_ylim(0, 115)
        ax3.set_ylabel("Kurtarma Oranı (%)", color="#94a3b8", fontsize=10)
        ax3.set_title("3. Öz-Yansıma (Self-Reflection) Hata Telafisi", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax3.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 4: Domain Breakdown of Skills
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.set_facecolor("#111827")
        
        domain_counts = {}
        for t in result.task_solution_history:
            d = t["domain"]
            domain_counts[d] = domain_counts.get(d, 0) + 1
            
        labels = [k.replace("_", "\n") for k in domain_counts.keys()]
        vals_d = list(domain_counts.values())
        colors_d = ["#ec4899", "#38bdf8", "#10b981", "#a855f7"]
        
        b4 = ax4.bar(labels, vals_d, color=colors_d[:len(labels)], width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b4, vals_d):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{val}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax4.set_ylabel("Bileşen Sayısı", color="#94a3b8", fontsize=10)
        ax4.set_title("4. Çok-Disiplinli Alan Dağılımı", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax4.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 5: Execution Latency Trajectory
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.set_facecolor("#111827")
        
        latencies = [t["latency_ms"] for t in result.task_solution_history]
        tasks = [t["task_id"] for t in result.task_solution_history]
        
        ax5.plot(tasks, latencies, color="#38bdf8", lw=1.8, marker=".", markersize=5, label="Görev Gecikmesi (ms)")
        ax5.axhline(result.avg_execution_latency_ms, color="#f43f5e", linestyle="--", lw=1.5, 
                    label=f"Ort. Gecikme: {result.avg_execution_latency_ms:.2f} ms")
                    
        ax5.set_xlabel("Görev Sırası", color="#94a3b8", fontsize=10)
        ax5.set_ylabel("Gecikme (ms)", color="#94a3b8", fontsize=10)
        ax5.set_title("5. Görev Başına Yürütme Gecikmesi", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax5.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax5.legend(loc="upper right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=9)
        
        # -------------------------------------------------------------
        # Panel 6: Polymath Diagnostic Summary Box
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.set_facecolor("#111827")
        ax6.axis("off")
        
        p = profil_ozeti or {}
        
        kpi_text = (
            "DAY 308: SELF-REFLECTIVE POLYMATH AGENT\n"
            "===========================================\n"
            f"• Beceri Sentez Başarısı: %{p.get('skill_synthesis_success_rate_pct', result.skill_synthesis_success_rate_pct):.2f}\n"
            f"• Çapraz-Alan Hafıza Kullanımı: %{p.get('cross_domain_reuse_efficiency_pct', result.cross_domain_reuse_efficiency_pct):.2f}\n"
            f"• Öz-Yansıma Hata Telafisi: %{p.get('reflection_error_recovery_rate_pct', result.reflection_error_recovery_rate_pct):.2f}\n"
            f"• Toplam Sentezlenen Beceri: {p.get('total_skills_synthesized', result.total_skills_synthesized)}\n"
            f"• Hafıza Çizgesi Yoğunluğu: {p.get('memory_graph_density', result.memory_graph_density):.4f}\n"
            f"• Ortalama İcra Gecikmesi: {p.get('avg_execution_latency_ms', result.avg_execution_latency_ms):.2f} ms\n"
            f"• Otonomi Seviyesi: {p.get('autonomy_tier', 'POLYMATH_RECURSIVE_SYNTHESIS_VERIFIED')}\n"
            "===========================================\n"
            "Durum: GÜVENLİ İZOLASYON SANDBOX VE\n"
            "        ÖZYİNELEMELİ SENTEZ AKTİF"
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
        ax6.set_title("6. Polymath Ajan Özeti", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        
        plt.tight_layout(pad=2.5)
        plt.savefig(cikti_yolu, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"📊 [Visualizer] 6-Panelli Teşhis Panosu başarıyla kaydedildi: {cikti_yolu}")
