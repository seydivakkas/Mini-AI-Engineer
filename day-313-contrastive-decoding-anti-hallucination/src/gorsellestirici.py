"""
Day 313: 6-Panel Diagnostic Dashboard Visualizer for Contrastive Decoding Anti-Hallucination.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, Optional
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .karsitsal_kod_cozucu import ContrastiveDecodingResult


class ContrastiveDecodingGorsellestirici:
    """
    Renders a 6-panel dark-mode diagnostic dashboard for Contrastive Decoding.
    """
    
    @staticmethod
    def ciz(result: ContrastiveDecodingResult, cikti_yolu: str = "ciktilar/karsitsal_kod_paneli.png", 
            profil_ozeti: Optional[Dict[str, Any]] = None):
        """
        Generates and saves the 6-panel diagnostic dashboard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)
        
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.patch.set_facecolor("#0b0f19")
        
        # -------------------------------------------------------------
        # Panel 1: Factuality Rate Comparison
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#111827")
        
        methods = ["Standart Çıkarım\n(Greedy Expert)", "Karşıtsal Kod Çözme\n(Contrastive Decoding)"]
        facts = [result.standard_factuality_pct, result.contrastive_factuality_pct]
        colors1 = ["#f43f5e", "#10b981"]
        
        b1 = ax1.bar(methods, facts, color=colors1, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b1, facts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"%{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax1.set_ylim(0, 115)
        ax1.set_ylabel("Olgusal Doğruluk Oranı (%)", color="#94a3b8", fontsize=10)
        ax1.set_title("1. Olgusal Doğruluk (Factuality Rate)", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax1.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 2: Step-by-Step Autoregressive Trajectory
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#111827")
        
        steps = np.arange(len(result.step_factuality_trajectory_std)) + 1
        ax2.plot(steps, result.step_factuality_trajectory_cd, color="#10b981", lw=2.5, marker="o", markersize=4, label="Contrastive Decoding")
        ax2.plot(steps, result.step_factuality_trajectory_std, color="#f43f5e", lw=2.0, linestyle="--", marker="x", markersize=4, label="Standart Greedy")
        
        ax2.set_xlabel("Üretim Adımı (Token Step)", color="#94a3b8", fontsize=10)
        ax2.set_ylabel("Adım Başı Doğruluk (%)", color="#94a3b8", fontsize=10)
        ax2.set_title("2. Adım Başı Olgusal Sapma Eğrisi", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax2.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax2.legend(loc="lower left", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
        
        # -------------------------------------------------------------
        # Panel 3: Hallucination Suppression Rate
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.set_facecolor("#111827")
        
        red = result.hallucination_reduction_pct
        labels = ["Baskılanan Halüsinasyon", "Kalan Hata"]
        sizes = [max(red, 0), max(100.0 - red, 0)]
        colors_pie = ["#38bdf8", "#334155"]
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors_pie,
                                            startangle=140, textprops=dict(color="#f8fafc", fontsize=9, fontweight="bold"),
                                            wedgeprops=dict(width=0.45, edgecolor="#0b0f19", lw=2))
        ax3.set_title("3. Halüsinasyon Azaltma Oranı", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        
        # -------------------------------------------------------------
        # Panel 4: Expected Calibration Error (ECE)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.set_facecolor("#111827")
        
        ece_methods = ["Standart\n(Aşırı Özgüven)", "Contrastive Decoding\n(Doğrulanmış Kalibrasyon)"]
        eces = [result.standard_ece, result.contrastive_ece]
        colors_ece = ["#f59e0b", "#10b981"]
        
        b4 = ax4.bar(ece_methods, eces, color=colors_ece, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b4, eces):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.4f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax4.set_ylabel("ECE Hatası (Düşük Daha İyi)", color="#94a3b8", fontsize=10)
        ax4.set_title("4. Güven Kalibrasyonu (Expected Calibration Error)", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax4.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 5: Sample Probe Prompt Accuracy
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.set_facecolor("#111827")
        
        prompts = [f"Prompt #{s['prompt_id']}" for s in result.sample_generations]
        std_p = [s["std_accuracy"] for s in result.sample_generations]
        cd_p = [s["cd_accuracy"] for s in result.sample_generations]
        
        x = np.arange(len(prompts))
        width = 0.35
        
        ax5.bar(x - width/2, std_p, width, label="Standart", color="#f43f5e", alpha=0.85, edgecolor="#ffffff", lw=1.0)
        ax5.bar(x + width/2, cd_p, width, label="Contrastive", color="#10b981", alpha=0.85, edgecolor="#ffffff", lw=1.0)
        
        ax5.set_xticks(x)
        ax5.set_xticklabels(prompts, color="#94a3b8", fontsize=9)
        ax5.set_ylim(0, 115)
        ax5.set_ylabel("İstem Doğruluğu (%)", color="#94a3b8", fontsize=10)
        ax5.set_title("5. Örnek İstem Başarı Dağılımı", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax5.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax5.legend(loc="lower right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
        
        # -------------------------------------------------------------
        # Panel 6: Telemetry Summary Box
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.set_facecolor("#111827")
        ax6.axis("off")
        
        p = profil_ozeti or {}
        
        kpi_text = (
            "DAY 313: CONTRASTIVE DECODING TELEMETRY\n"
            "==============================================\n"
            f"• Standart Olgusal Doğruluk: %{p.get('standard_factuality_pct', result.standard_factuality_pct):.2f}\n"
            f"• Karşıtsal Olgusal Doğruluk: %{p.get('contrastive_factuality_pct', result.contrastive_factuality_pct):.2f}\n"
            f"• Halüsinasyon Azaltma: %{p.get('hallucination_reduction_pct', result.hallucination_reduction_pct):.2f}\n"
            f"• Standart ECE: {p.get('standard_ece', result.standard_ece):.4f}\n"
            f"• Karşıtsal ECE: {p.get('contrastive_ece', result.contrastive_ece):.4f}\n"
            f"• Toplam Üretilen Token: {result.tokens_generated:,}\n"
            f"• Kalibrasyon Sınıfı: {p.get('calibration_tier', 'HIGH_PRECISION_GROUNDED')}\n"
            "==============================================\n"
            "DURUM: AMATÖR LOGİT CEZALANDIRMASI VE\n"
            "        UYARLANABİLİR BAŞLIK TRUNCATION AKTİF"
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
        ax6.set_title("6. Karşıtsal Kod Çözme Modeli Özeti", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        
        plt.tight_layout(pad=2.5)
        plt.savefig(cikti_yolu, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"📊 [Visualizer] 6-Panelli Teşhis Panosu başarıyla kaydedildi: {cikti_yolu}")
