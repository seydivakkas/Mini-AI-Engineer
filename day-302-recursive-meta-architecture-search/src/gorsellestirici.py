"""
Day 302: 6-Panel Diagnostic Dashboard Visualizer for Meta-NAS.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, Optional
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .meta_nas_motoru import NASSearchResult, OP_NAMES


class MetaNASGorsellestirici:
    """
    Renders a high-resolution, 6-panel dark-mode diagnostic visualization for Meta-NAS.
    """
    
    @staticmethod
    def ciz(result: NASSearchResult, cikti_yolu: str = "ciktilar/meta_nas_paneli.png", 
            profil_ozeti: Optional[Dict[str, Any]] = None):
        """
        Generates and saves the 6-panel diagnostic dashboard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)
        
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.patch.set_facecolor("#0b0f19")
        
        history = result.search_history
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        
        # -------------------------------------------------------------
        # Panel 1: Bi-Level Loss Dynamics (Train vs Val)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#111827")
        ax1.plot(epochs, history["train_loss"], color="#38bdf8", lw=2.2, label="Inner Loop (Train Loss)")
        ax1.plot(epochs, history["val_loss"], color="#f43f5e", lw=2.2, linestyle="--", label="Outer Loop (Val Loss)")
        ax1.set_title("1. Bi-Level Optimizasyon Kayıp Eğrileri", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax1.set_xlabel("Arama Epoku", color="#94a3b8", fontsize=10)
        ax1.set_ylabel("Cross-Entropy Kaybı", color="#94a3b8", fontsize=10)
        ax1.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155")
        
        # -------------------------------------------------------------
        # Panel 2: Architecture Alphas Heatmap (Edges x Operations)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#111827")
        
        # Softmax probabilities of final alpha
        alpha_matrix = result.supernet_alpha
        exp_alpha = np.exp(alpha_matrix - np.max(alpha_matrix, axis=-1, keepdims=True))
        probs_matrix = exp_alpha / np.sum(exp_alpha, axis=-1, keepdims=True)
        
        im = ax2.imshow(probs_matrix, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
        ax2.set_title("2. Süpernet Mimari Ağırlıkları P(α)", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel("Aday Operasyonlar", color="#94a3b8", fontsize=10)
        ax2.set_ylabel("DAG Kenar İndeksi (i → j)", color="#94a3b8", fontsize=10)
        ax2.set_xticks(range(len(OP_NAMES)))
        ax2.set_xticklabels(OP_NAMES, rotation=45, ha="right", color="#cbd5e1", fontsize=9)
        ax2.set_yticks(range(alpha_matrix.shape[0]))
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="#94a3b8")
        
        # -------------------------------------------------------------
        # Panel 3: Temperature Annealing & Entropy Reduction
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.set_facecolor("#111827")
        color_tau = "#a855f7"
        color_ent = "#10b981"
        
        ax3.plot(epochs, history["temperature"], color=color_tau, lw=2.2, label="Sıcaklık τ(t)")
        ax3.set_xlabel("Arama Epoku", color="#94a3b8", fontsize=10)
        ax3.set_ylabel("Gumbel Sıcaklığı (τ)", color=color_tau, fontsize=10)
        ax3.tick_params(axis="y", labelcolor=color_tau)
        
        ax3_twin = ax3.twinx()
        ax3_twin.plot(epochs, history["alpha_entropy"], color=color_ent, lw=2.2, linestyle="-.", label="Mimari Entropisi H(α)")
        ax3_twin.set_ylabel("Entropi (Nat)", color=color_ent, fontsize=10)
        ax3_twin.tick_params(axis="y", labelcolor=color_ent)
        
        ax3.set_title("3. Sıcaklık Azaltma ve Entropi Çöküşü", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax3.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 4: Multi-Objective Pareto Frontier (Acc vs FLOPs)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.set_facecolor("#111827")
        
        all_cands = result.all_candidates
        all_flops = [c.flops_m for c in all_cands]
        all_accs = [c.accuracy for c in all_cands]
        all_lats = [c.latency_ms for c in all_cands]
        
        scatter = ax4.scatter(all_flops, all_accs, c=all_lats, cmap="cool", s=65, alpha=0.7, edgecolors="none", label="Örneklenen Adaylar")
        
        # Highlight Pareto Frontier
        pareto_cands = result.pareto_frontier
        p_flops = [c.flops_m for c in pareto_cands]
        p_accs = [c.accuracy for c in pareto_cands]
        
        # Sort for step line
        sorted_p = sorted(zip(p_flops, p_accs), key=lambda x: x[0])
        sp_f, sp_a = zip(*sorted_p)
        ax4.plot(sp_f, sp_a, color="#fbbf24", lw=2.0, linestyle="--", marker="o", markersize=8, label="Pareto Optimum Sınırı")
        
        # Highlight Best
        best = result.best_candidate
        ax4.scatter([best.flops_m], [best.accuracy], color="#ef4444", s=180, marker="*", edgecolors="#ffffff", lw=1.5, zorder=5, label="Seçilen En İyi")
        
        ax4.set_title("4. Çok Amaçlı Pareto Optimum Dengesi", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax4.set_xlabel("Hesaplama Maliyeti (MFLOPs)", color="#94a3b8", fontsize=10)
        ax4.set_ylabel("Doğruluk Başarımı (%)", color="#94a3b8", fontsize=10)
        ax4.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        cbar4 = fig.colorbar(scatter, ax=ax4, fraction=0.046, pad=0.04)
        cbar4.set_label("Gecikme (ms)", color="#94a3b8", fontsize=9)
        cbar4.ax.tick_params(colors="#94a3b8")
        ax4.legend(loc="lower right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=9)
        
        # -------------------------------------------------------------
        # Panel 5: Discretization Gap & Candidate Distribution
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.set_facecolor("#111827")
        
        cand_ids = np.arange(len(all_cands))
        cand_accs = [c.accuracy for c in all_cands]
        bar_colors = ["#fbbf24" if c in pareto_cands else "#475569" for c in all_cands]
        
        bars = ax5.bar(cand_ids, cand_accs, color=bar_colors, alpha=0.85, width=0.6)
        
        # Supernet continuous accuracy line
        final_cont_acc = history["val_acc"][-1]
        ax5.axhline(final_cont_acc, color="#38bdf8", linestyle="--", lw=1.8, label=f"Sürekli Süpernet (%{final_cont_acc:.1f})")
        
        ax5.set_title("5. Ayrıklaştırma Farkı & Aday Dağılımı", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax5.set_xlabel("Aday Mimari İndeksi", color="#94a3b8", fontsize=10)
        ax5.set_ylabel("Doğruluk (%)", color="#94a3b8", fontsize=10)
        ax5.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax5.legend(loc="lower right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=9)
        
        # -------------------------------------------------------------
        # Panel 6: Diagnostic & Performance Summary KPI Box
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.set_facecolor("#111827")
        ax6.axis("off")
        
        p = profil_ozeti or {}
        
        kpi_text = (
            "DAY 302: META-NAS SEARCH SUMMARY\n"
            "===========================================\n"
            f"• En Iyi Aday Dogrulugu: %{p.get('best_cand_acc', best.accuracy):.2f}\n"
            f"• Supernet Son Val Dogruluk: %{p.get('final_val_acc', final_cont_acc):.2f}\n"
            f"• Ayriklastirma Farki (Delta_disc): %{p.get('discretization_gap', 0.0):.2f}\n"
            f"• Secilen Model FLOPs: {p.get('best_cand_flops_m', best.flops_m):.3f} MFLOPs\n"
            f"• FLOPs Sikistirma Kazanci: %{p.get('flops_reduction_pct', 0.0):.1f}\n"
            f"• Cikarim Gecikmesi: {p.get('best_cand_latency_ms', best.latency_ms):.2f} ms\n"
            f"• Gecikme Hizlanmasi: {p.get('latency_speedup_x', 0.0):.2f}x Hizli\n"
            f"• Pareto Hiper-Hacim Skoru: {p.get('pareto_hypervolume_score', 0.0):.1f} / 100\n"
            f"• Entropi Azalmasi: %{p.get('entropy_reduction_pct', 0.0):.1f}\n"
            f"• Pareto Optimum Aday Sayisi: {p.get('pareto_frontier_count', len(pareto_cands))}\n"
            f"• Toplam Arama Suresi: {p.get('search_time_sec', result.search_time_sec):.2f} saniye\n"
            "===========================================\n"
            f"[Gen]: {', '.join(best.gene[:3])}...\n"
            "Durum: META-MIMARI OPTIMUM KESFEDILDI"
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
        ax6.set_title("6. Teşhis ve Donanım Verimlilik Özeti", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        
        plt.tight_layout(pad=2.5)
        plt.savefig(cikti_yolu, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"📊 [Visualizer] 6-Panelli Teşhis Panosu başarıyla kaydedildi: {cikti_yolu}")
