"""
Pipeline Parallelism 1F1B 6 Panelli Görselleştirici Modülü (Day 185 - FAZ 10).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class PipelineGorsellestirici:
    """Pipeline Parallelism 1F1B 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def pipeline_teshis_paneli_olustur(
        cls,
        cizelge_raporu: List[Dict[str, Any]],
        kayit_yolu: str = "ciktilar/pipeline_parallelism_1f1b_paneli.png",
    ):
        """6 Panelli Pipeline Parallelism Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 185: PIPELINE PARALLELISM (PP) — 1F1B ZAMAN ÇİZELGESİ VE BALON ANALİZİ",
            fontsize=18,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: GPipe vs 1F1B Zaman Çizelgesi (Gantt Simülasyonu)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        asama_adlari = ["Stage 3", "Stage 2", "Stage 1", "Stage 0"]
        # 1F1B için Forward (Mavi), Backward (Yeşil), Bubble (Koyu Gri)
        # Stage 0: F0, F1, F2, F3, B0, F4, B1, F5, B2...
        b_tasks = [
            (0, 1, "#3b82f6"), (1, 1, "#3b82f6"), (2, 1, "#3b82f6"), (3, 1, "#3b82f6"),
            (4, 1, "#10b981"), (5, 1, "#3b82f6"), (6, 1, "#10b981"), (7, 1, "#3b82f6"),
        ]
        for start, dur, col in b_tasks:
            ax1.barh(3, dur, left=start, color=col, edgecolor="#ffffff", height=0.55)

        # Stage 1: Idle 1, F0, F1, F2, B0, F3, B1...
        s1_tasks = [
            (1, 1, "#3b82f6"), (2, 1, "#3b82f6"), (3, 1, "#3b82f6"), (4, 1, "#3b82f6"),
            (5, 1, "#10b981"), (6, 1, "#3b82f6"), (7, 1, "#10b981"),
        ]
        for start, dur, col in s1_tasks:
            ax1.barh(2, dur, left=start, color=col, edgecolor="#ffffff", height=0.55)

        # Stage 2: Idle 2, F0, F1, F2, B0...
        s2_tasks = [(2, 1, "#3b82f6"), (3, 1, "#3b82f6"), (4, 1, "#3b82f6"), (5, 1, "#10b981"), (6, 1, "#3b82f6"), (7, 1, "#10b981")]
        for start, dur, col in s2_tasks:
            ax1.barh(1, dur, left=start, color=col, edgecolor="#ffffff", height=0.55)

        # Stage 3: Idle 3, F0, F1, B0, B1...
        s3_tasks = [(3, 1, "#3b82f6"), (4, 1, "#10b981"), (5, 1, "#3b82f6"), (6, 1, "#10b981"), (7, 1, "#10b981")]
        for start, dur, col in s3_tasks:
            ax1.barh(0, dur, left=start, color=col, edgecolor="#ffffff", height=0.55)

        ax1.set_yticks(range(4))
        ax1.set_yticklabels(asama_adlari, fontsize=9)
        ax1.set_xlabel("Zaman Adımları (Time Steps)", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. 1F1B Zaman Çizelgesi Gantt İcrası (P=4 Stage)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: Pipeline Bubble Oranı vs Micro-batch Sayısı (M)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        m_vals = np.array([4, 8, 16, 32, 64, 128])
        p_stages = [4, 8, 16]
        colors = ["#38bdf8", "#10b981", "#f59e0b"]

        for p, c in zip(p_stages, colors):
            bubbles = [((p - 1) / (m + p - 1)) * 100.0 for m in m_vals]
            ax2.plot(m_vals, bubbles, marker="o", color=c, linewidth=2, label=f"P = {p} Aşama")

        ax2.set_xlabel("Mikro-Batch Sayısı (M)", fontsize=10, color="#cbd5e1")
        ax2.set_ylabel("Balon (Bubble / Idle) Oranı (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Balon Oranı vs Mikro-Batch Sayısı ($F_{bubble} = \\frac{P-1}{M+P-1}$)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 3: Tepe Aktivasyon Belleği (GPipe vs 1F1B)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        cizelgeler = [r["cizelge_adi"].split("(")[0].strip() for r in cizelge_raporu]
        vram_vals = [r["tepe_aktivasyon_gb"] for r in cizelge_raporu]
        bar_colors = ["#ef4444", "#22c55e", "#0284c7"]

        bars3 = ax3.bar(cizelgeler, vram_vals, color=bar_colors, width=0.45)
        ax3.set_ylabel("Tepe Aktivasyon Belleği (GB)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Tepe Aktivasyon Belleği: GPipe O(M) vs 1F1B O(P)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.grid(axis="y", linestyle=":", alpha=0.4)
        for b in bars3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 0.1, f"{h:.2f} GB", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9)

        # -------------------------------------------------------------
        # PANEL 4: Interleaved 1F1B Balon Azaltma (Sanal Aşamalar v)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        v_degerleri = [1, 2, 4]
        v_adlar = ["Standart 1F1B\n(v=1)", "Interleaved 1F1B\n(v=2 Sanal Aşama)", "Interleaved 1F1B\n(v=4 Sanal Aşama)"]
        m_sabit = 32
        p_sabit = 8
        balon_yuzdeleri = [
            ((p_sabit - 1) / (m_sabit + p_sabit - 1)) * 100.0,
            ((p_sabit - 1) / (2 * m_sabit)) * 100.0,
            ((p_sabit - 1) / (4 * m_sabit)) * 100.0,
        ]

        bars4 = ax4.bar(v_adlar, balon_yuzdeleri, color=["#ef4444", "#3b82f6", "#10b981"], width=0.5)
        ax4.set_ylabel("Balon Kesri (%)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Sanal Aşamalarla (v) Balon Azaltımı ($P=8, M=32$)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.grid(axis="y", linestyle=":", alpha=0.4)
        for b in bars4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 0.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9)

        # -------------------------------------------------------------
        # PANEL 5: P2P Noktadan Noktaya İletişim İzi
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        asamalar_p2p = ["Aşama 0 -> 1", "Aşama 1 -> 2", "Aşama 2 -> 3", "Aşama 3 -> 2 (Bwd)", "Aşama 2 -> 1 (Bwd)", "Aşama 1 -> 0 (Bwd)"]
        transfer_boyut_mb = [12.5, 12.5, 12.5, 12.5, 12.5, 12.5]  # Mikro-batch aktivasyon boyutu

        ax5.barh(asamalar_p2p, transfer_boyut_mb, color="#0284c7", edgecolor="#38bdf8", height=0.5)
        ax5.set_xlabel("P2P Transfer Hacmi (MB / Micro-Batch)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. P2P Aşama Sınırı İletişim Hacmi (Düşük Ağ Yükü)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 6: GÜN 185 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "GÜN 185: PIPELINE PARALLELISM 1F1B KARNE\n"
            "----------------------------------------------------\n"
            "• Bölümleme Türü      : Katman Boyutunda (Stage 0..P-1)\n"
            "• İletişim Türü       : Yalnızca Komşu P2P (Send/Recv)\n"
            "• Ağ Gereksinimi      : Düşük Bant Genişliği (Inter-Node)\n"
            "• Zaman Çizelgesi     : 1F1B (Warmup -> 1F1B -> Cooldown)\n"
            "• Bellek Kazancı      : O(M) -> O(P) (%75+ VRAM Tasarrufu)\n"
            "• Balon Optimizasyonu : Interleaved 1F1B (v=2 ile %50 balon kesimi)\n"
            "• Balon Formülü       : F_bubble = (P-1) / (v * M)\n"
            "----------------------------------------------------\n"
            "SONUÇ: Yüzlerce katmanlı 100B+ modelleri farklı sunuculara\n"
            "(Nodlar Arası) dağıtmanın en verimli paralel mimarisi!"
        )

        ax6.text(
            0.05,
            0.5,
            ozet_metin,
            fontsize=10,
            family="monospace",
            color="#f8fafc",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.9),
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close()
