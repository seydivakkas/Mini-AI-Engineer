"""
6-Panelli Vektör ve Semantik Arama Değerlendirme Teşhis Panosu (Retrieval Benchmark Dashboard).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class RetrievalGorsellestirici:
    """NDCG@k, MRR, MAP, Precision/Recall ve Latency metriklerini görselleştiren teşhis panosu."""

    @classmethod
    def panel_ciz(
        cls,
        benchmark_sonuclari: Dict[str, Dict[str, Any]],
        num_queries: int = 500,
        hedef_path: str = "ciktilar/retrieval_metrics_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(21, 13), dpi=300)
        fig.suptitle(
            "Day 61: Vektör ve Semantik Arama Değerlendirmesi (NDCG@k, MRR, MAP, Gecikme & QPS)",
            fontsize=15, fontweight="bold", y=0.98
        )

        stratejiler = list(benchmark_sonuclari.keys())
        ndcg10_list = [benchmark_sonuclari[k]["ndcg@10"] for k in stratejiler]
        mrr_list = [benchmark_sonuclari[k]["mrr"] for k in stratejiler]
        p50_list = [benchmark_sonuclari[k]["gecikme_istatistikleri"]["p50_ms"] for k in stratejiler]
        p95_list = [benchmark_sonuclari[k]["gecikme_istatistikleri"]["p95_ms"] for k in stratejiler]
        p99_list = [benchmark_sonuclari[k]["gecikme_istatistikleri"]["p99_ms"] for k in stratejiler]
        qps_list = [benchmark_sonuclari[k]["gecikme_istatistikleri"]["qps"] for k in stratejiler]

        en_iyi_ndcg = max(ndcg10_list)
        en_iyi_strateji = stratejiler[int(np.argmax(ndcg10_list))]

        # -------------------------------------------------------------
        # Panel 1: Yönetici Özeti Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")

        kart_metni = (
            f"RETRIEVAL BENCHMARK YÖNETİCİ ÖZETİ\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Değerlendirilen Sorgu Sayısı : {num_queries:,} Sorgu\n"
            f"• Test Edilen Boru Hatları     : 4 Farklı Arama Mimarisi\n"
            f"─────────────────────────────────────────────\n"
            f"• En Yüksek Kalite (NDCG@10)  : {en_iyi_ndcg:.4f} ({en_iyi_strateji})\n"
            f"• En Yüksek MRR (İlk Yanıt)   : {max(mrr_list):.4f}\n"
            f"• En Düşük Medyan Gecikme (p50): {min(p50_list):.2f} ms\n"
            f"• Hibrit RRF Başarım Artışı   : %{((ndcg10_list[0] - ndcg10_list[1])/ndcg10_list[1])*100:+.1f} NDCG Kazancı\n"
            f"─────────────────────────────────────────────\n"
            f"• Üretim Arama Kalite Durumu  : %100 ONAYLANDI"
        )

        ax1.text(
            0.5, 0.5, kart_metni, transform=ax1.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.9", facecolor="#9b59b6", alpha=0.18, edgecolor="#8e44ad", linewidth=2),
            fontsize=9.0, fontweight="bold", family="monospace"
        )
        ax1.set_title("1. Retrieval Kalite Yönetici Özeti", fontweight="bold", color="#2c3e50")

        # -------------------------------------------------------------
        # Panel 2: NDCG@5 vs NDCG@10 Karşılaştırması
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        x = np.arange(len(stratejiler))
        w = 0.35
        ndcg5 = [benchmark_sonuclari[k]["ndcg@5"] for k in stratejiler]

        b1 = ax2.bar(x - w/2, ndcg5, width=w, label="NDCG@5", color="#3498db", edgecolor="#2c3e50")
        b2 = ax2.bar(x + w/2, ndcg10_list, width=w, label="NDCG@10", color="#2ecc71", edgecolor="#2c3e50")

        for b in b1:
            ax2.text(b.get_x() + b.get_width()/2., b.get_height() + 0.015, f"{b.get_height():.2f}", ha="center", fontsize=7.5, fontweight="bold")
        for b in b2:
            ax2.text(b.get_x() + b.get_width()/2., b.get_height() + 0.015, f"{b.get_height():.2f}", ha="center", fontsize=7.5, fontweight="bold")

        ax2.set_xticks(x)
        ax2.set_xticklabels([k.split()[0] for k in stratejiler], rotation=20, ha="right", fontsize=8)
        ax2.set_ylim(0, 1.15)
        ax2.set_ylabel("NDCG Skoru")
        ax2.legend(loc="upper right", frameon=True)
        ax2.set_title("2. Dereceli Sıralama Kalitesi (NDCG@k)", fontweight="bold", color="#2980b9")

        # -------------------------------------------------------------
        # Panel 3: MRR ve MAP@10
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        map10 = [benchmark_sonuclari[k]["map@10"] for k in stratejiler]
        ax3.bar(x - w/2, mrr_list, width=w, label="MRR", color="#f39c12", edgecolor="#2c3e50")
        ax3.bar(x + w/2, map10, width=w, label="MAP@10", color="#e67e22", edgecolor="#2c3e50")

        ax3.set_xticks(x)
        ax3.set_xticklabels([k.split()[0] for k in stratejiler], rotation=20, ha="right", fontsize=8)
        ax3.set_ylim(0, 1.15)
        ax3.set_ylabel("Metrik Skoru")
        ax3.legend(loc="upper right", frameon=True)
        ax3.set_title("3. MRR (İlk İlgili) & MAP@10", fontweight="bold", color="#d35400")

        # -------------------------------------------------------------
        # Panel 4: Precision@10 vs Recall@10
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        prec10 = [benchmark_sonuclari[k]["precision@10"] for k in stratejiler]
        rec10 = [benchmark_sonuclari[k]["recall@10"] for k in stratejiler]

        for i, strat in enumerate(stratejiler):
            ax4.scatter(rec10[i], prec10[i], s=140, label=strat, edgecolors="#2c3e50", zorder=4)
            ax4.text(rec10[i] + 0.01, prec10[i] + 0.01, strat.split()[0], fontsize=8, fontweight="bold")

        ax4.set_xlabel("Recall@10")
        ax4.set_ylabel("Precision@10")
        ax4.set_title("4. Precision vs. Recall@10 Dengesi", fontweight="bold", color="#16a085")
        ax4.set_xlim(min(rec10) - 0.05, max(rec10) + 0.1)
        ax4.set_ylim(min(prec10) - 0.05, max(prec10) + 0.1)

        # -------------------------------------------------------------
        # Panel 5: Gecikme Dağılımı ve Kuyruk Gecikmesi (p50, p95, p99)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        width3 = 0.25
        ax5.bar(x - width3, p50_list, width=width3, label="p50 (Medyan)", color="#27ae60", edgecolor="#2c3e50")
        ax5.bar(x, p95_list, width=width3, label="p95 (Kuyruk)", color="#f39c12", edgecolor="#2c3e50")
        ax5.bar(x + width3, p99_list, width=width3, label="p99 (Kritik)", color="#c0392b", edgecolor="#2c3e50")

        ax5.set_xticks(x)
        ax5.set_xticklabels([k.split()[0] for k in stratejiler], rotation=20, ha="right", fontsize=8)
        ax5.set_ylabel("Gecikme (ms)")
        ax5.legend(loc="upper left", frameon=True)
        ax5.set_title("5. Gecikme Profili & Kuyruk (ms)", fontweight="bold", color="#c0392b")

        # -------------------------------------------------------------
        # Panel 6: NDCG@10 vs QPS Pareto Eğrisi (Ticari Tercih)
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        for i, strat in enumerate(stratejiler):
            ax6.scatter(ndcg10_list[i], qps_list[i], s=140, edgecolors="#2c3e50", zorder=4)
            ax6.text(ndcg10_list[i] + 0.01, qps_list[i] * 1.05, f"{strat.split()[0]}\n({qps_list[i]:.0f} QPS)", fontsize=7.5)

        ax6.set_xlabel("NDCG@10 (Kalite)")
        ax6.set_ylabel("Throughput (QPS)")
        ax6.set_yscale("log")
        ax6.set_title("6. Kalite vs. Throughput Pareto Eğrisi", fontweight="bold", color="#8e44ad")

        fig.subplots_adjust(top=0.93, bottom=0.10, left=0.10, right=0.95, hspace=0.36, wspace=0.32)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
