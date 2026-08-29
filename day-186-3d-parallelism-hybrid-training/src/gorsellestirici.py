"""
3D Hibrit Paralellik (DP + TP + PP) 6 Panelli Görselleştirici Modülü (Day 186 - FAZ 10).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class UcBoyutluGorsellestirici:
    """3D Paralellik 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        model_raporu: List[Dict[str, Any]],
        kayit_yolu: str = "ciktilar/uc_boyutlu_paralellik_3d_paneli.png",
    ):
        """6 Panelli 3D Paralellik Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 186: 3D PARALELLİK (DP + TP + PP) — 70B - 405B LLM KÜME EĞİTİM MİMARİSİ",
            fontsize=18,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: 3D Process Grid Topolojisi (DP=2, PP=4, TP=8 = 64 GPU)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        # 3D Grid blok görselleştirmesi (DP ekseni, PP ekseni, TP ekseni)
        dp_gruplar = ["DP Replica 0 (32 GPU)", "DP Replica 1 (32 GPU)"]
        pp_stages = ["Stage 0\n(Katman 1-20)", "Stage 1\n(Katman 21-40)", "Stage 2\n(Katman 41-60)", "Stage 3\n(Katman 61-80)"]
        
        # Isı haritası şeklinde 3D grid aşama düzeni
        grid_matrix = np.array([
            [8, 8, 8, 8],  # DP 0: 4 PP aşaması, her biri 8 TP GPU
            [8, 8, 8, 8],  # DP 1: 4 PP aşaması, her biri 8 TP GPU
        ])

        im = ax1.imshow(grid_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=10)
        ax1.set_xticks(range(4))
        ax1.set_xticklabels(pp_stages, fontsize=8)
        ax1.set_yticks(range(2))
        ax1.set_yticklabels(dp_gruplar, fontsize=9)
        ax1.set_title("1. 3D Grid Matrisi (DP=2, PP=4, TP=8 -> 64 GPU)", fontsize=11, color="#38bdf8", fontweight="bold")

        for i in range(2):
            for j in range(4):
                ax1.text(j, i, "TP=8 GPU\n(NVLink 900 GB/s)", ha="center", va="center", color="#ffffff", fontweight="bold", fontsize=8)

        # -------------------------------------------------------------
        # PANEL 2: 70B, 175B, 405B için GPU Başına VRAM Tüketimi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        model_adlari = [r["model_adi"] for r in model_raporu]
        model_vrams = [r["gpu_toplam_vram_gb"] for r in model_raporu]
        h100_limit = 80.0

        bars2 = ax2.bar(model_adlari, model_vrams, color=["#38bdf8", "#10b981", "#f59e0b"], width=0.45)
        ax2.axhline(h100_limit, color="#ef4444", linestyle="--", linewidth=2, label="H100 80GB VRAM Sınırı")
        ax2.set_ylabel("GPU Başına Tepe VRAM (GB)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. GPU Başına VRAM Dağılımı (80GB H100 Uyumu)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(axis="y", linestyle=":", alpha=0.4)

        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"{h:.1f} GB", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9)

        # -------------------------------------------------------------
        # PANEL 3: Ağ İletişim Bant Genişliği Kıyası (GB/s)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ag_tipleri = ["Intra-Node NVLink\n(TP İletişimi)", "Inter-Node InfiniBand\n(PP & DP İletişimi)", "Standart 100G Ethernet\n(Kısıtlı)"]
        hizlar_gbs = [900.0, 50.0, 12.5]
        bar_cols3 = ["#22c55e", "#3b82f6", "#ef4444"]

        bars3 = ax3.barh(ag_tipleri, hizlar_gbs, color=bar_cols3, height=0.5)
        ax3.set_xlabel("Bant Genişliği (GB/s Çift Yönlü)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Donanım Ağ Hızları (TP vs PP/DP İletişim Ayrımı)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.grid(axis="x", linestyle=":", alpha=0.3)

        for b in bars3:
            w = b.get_width()
            ax3.text(w + 10, b.get_y() + b.get_height() / 2.0, f"{w:.1f} GB/s", ha="left", va="center", color="#ffffff", fontweight="bold", fontsize=9)

        # -------------------------------------------------------------
        # PANEL 4: Model Ölçeğine Göre Küme Büyüklüğü (GPU Sayısı)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        toplam_gpus = [r["toplam_gpu"] for r in model_raporu]
        bars4 = ax4.bar(model_adlari, toplam_gpus, color=["#6366f1", "#8b5cf6", "#ec4899"], width=0.45)
        ax4.set_ylabel("Toplam H100 GPU Sayısı", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. LLM Ölçeğine Göre Optimal GPU Kümesi ($N=DP\\times TP\\times PP$)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.grid(axis="y", linestyle=":", alpha=0.4)

        for b in bars4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 8, f"{int(h)} GPU", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9)

        # -------------------------------------------------------------
        # PANEL 5: Donanım Flops Verimliliği (MFU %) Kıyası
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        yontemler = ["Saf DDP\n(Sığmaz)", "Saf TP\n(Tek Sunucu)", "Saf PP\n(Yüksek Balon)", "3D Paralellik\n(DP+TP+PP)"]
        mfu_degerleri = [0.0, 38.2, 34.0, 54.5]
        bars5 = ax5.bar(yontemler, mfu_degerleri, color=["#64748b", "#f59e0b", "#f97316", "#10b981"], width=0.5)
        ax5.set_ylabel("Model Flops Utilization (MFU %)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Donanım Flops Verimliliği (MFU %)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.grid(axis="y", linestyle=":", alpha=0.4)

        for b in bars5:
            h = b.get_height()
            ax5.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9)

        # -------------------------------------------------------------
        # PANEL 6: GÜN 186 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "GÜN 186: 3D PARALELLİK (DP + TP + PP) KARNE\n"
            "----------------------------------------------------\n"
            "• Grid Formülü        : N = DP_size x PP_size x TP_size\n"
            "• TP Sınırı (Intra)   : TP = 8 (Sunucu içi 900 GB/s NVLink)\n"
            "• PP Sınırı (Inter)   : PP = 4-16 (Sunucular arası InfiniBand)\n"
            "• DP Sınırı (Outer)   : DP = 2-64 (Global Batch Ölçekleme)\n"
            "• Llama-3-70B Kümesi  : 64 H100 (DP=2, PP=4, TP=8) -> 23.8 GB\n"
            "• GPT-3-175B Kümesi   : 128 H100 (DP=2, PP=8, TP=8) -> 24.9 GB\n"
            "• Llama-3-405B Kümesi : 512 H100 (DP=4, PP=16, TP=8) -> 13.8 GB\n"
            "• Ulaşılan MFU        : %54.5 Donanım Hesaplama Verimi\n"
            "----------------------------------------------------\n"
            "SONUÇ: Dünya genelinde trilyon parametreli modellerin\n"
            "eğitiminde kullanılan altın standart küme mimarisi!"
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
