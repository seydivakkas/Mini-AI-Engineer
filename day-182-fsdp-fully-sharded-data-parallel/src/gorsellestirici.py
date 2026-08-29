"""
FSDP 6 Panelli Teşhis ve Bellek Görselleştirici Modülü (Day 182 - FAZ 10).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class FSDPGorsellestirici:
    """FSDP (Fully Sharded Data Parallel) 6 Panelli Görselleştirme Motoru."""

    @classmethod
    def fsdp_teshis_paneli_olustur(
        cls,
        bellek_karsilastirma: List[Dict[str, Any]],
        layer_stats: List[Dict[str, Any]],
        kayit_yolu: str = "ciktilar/fsdp_fully_sharded_data_parallel_paneli.png",
    ):
        """6 Panelli FSDP Teşhis ve Bellek Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 182: FULLY SHARDED DATA PARALLEL (FSDP) & ZeRO-3 BELLEK TEŞHİS PANOSU",
            fontsize=18,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: DDP vs ZeRO-1 vs ZeRO-2 vs FSDP VRAM Tüketimi (70B Model)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        yontemler = ["DDP (ZeRO-0)", "ZeRO-1 (Opt)", "ZeRO-2 (Grad+Opt)", "FSDP (ZeRO-3)"]
        # 70B model 64 GPU VRAM bileşenleri (GB)
        # DDP: Param 140GB, Grad 140GB, Opt 840GB = 1120GB
        # ZeRO-1: Param 140GB, Grad 140GB, Opt 13.1GB = 293.1GB
        # ZeRO-2: Param 140GB, Grad 2.2GB, Opt 13.1GB = 155.3GB
        # FSDP: Param 2.2GB, Grad 2.2GB, Opt 13.1GB = 17.5GB
        param_gb = [140.0, 140.0, 140.0, 2.2]
        grad_gb = [140.0, 140.0, 2.2, 2.2]
        opt_gb = [840.0, 13.1, 13.1, 13.1]

        x_pos = np.arange(len(yontemler))
        ax1.bar(x_pos, param_gb, label="Parametreler (FP16)", color="#3b82f6", width=0.55)
        ax1.bar(x_pos, grad_gb, bottom=param_gb, label="Gradyanlar (FP16)", color="#10b981", width=0.55)
        ax1.bar(x_pos, opt_gb, bottom=np.array(param_gb) + np.array(grad_gb), label="AdamW Opt (FP32)", color="#f59e0b", width=0.55)

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(yontemler, fontsize=9, rotation=15)
        ax1.set_ylabel("GPU Başına Bellek (GB)", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. 70B Model İçin Statik VRAM Kıyası (64 GPU)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.axhline(80, color="#ef4444", linestyle="--", alpha=0.7, label="A100/H100 80GB Limiti")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(axis="y", linestyle=":", alpha=0.4)

        # -------------------------------------------------------------
        # PANEL 2: Katman Bazlı FSDP İletişim & Bellek Döngüsü
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        asamalar = ["1. All-Gather\n(Layer $L$)", "2. Forward\nCompute", "3. Free Weights\n(Drop)", "4. Backward\nAll-Gather", "5. Reduce-Scatter\n(Gradients)"]
        vram_seviyeleri = [100, 100, 25, 100, 25]  # VRAM kullanım yüzdesi (Normalized)
        renkler = ["#0284c7", "#10b981", "#64748b", "#0284c7", "#f59e0b"]

        bars = ax2.bar(asamalar, vram_seviyeleri, color=renkler, width=0.5)
        ax2.set_ylabel("Geçici VRAM Seviyesi (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. FSDP Katman Yaşam Döngüsü & Bellek Tepe Noktaları", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.grid(axis="y", linestyle=":", alpha=0.4)
        for b in bars:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 2, f"%{h}", ha="center", va="bottom", color="#f8fafc", fontsize=9, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 3: Model Parametre Boyutu vs VRAM (64 GPU)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        modeller = [m["model_adi"] for m in bellek_karsilastirma]
        ddp_vals = [m["ddp_gb"] for m in bellek_karsilastirma]
        fsdp_vals = [m["fsdp_gb"] for m in bellek_karsilastirma]

        x = np.arange(len(modeller))
        width = 0.35
        ax3.bar(x - width / 2, ddp_vals, width, label="Standart DDP", color="#ef4444")
        ax3.bar(x + width / 2, fsdp_vals, width, label="FSDP (ZeRO-3)", color="#22c55e")

        ax3.set_yscale("log")
        ax3.set_xticks(x)
        ax3.set_xticklabels(modeller, fontsize=9)
        ax3.set_ylabel("Bellek GB (Log Ölçek)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Model Boyutlarına Göre DDP vs FSDP (Log Ölçek)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.axhline(80, color="#f59e0b", linestyle=":", label="80 GB GPU VRAM")
        ax3.legend(loc="upper left", fontsize=8)
        ax3.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 4: Backward Prefetch & İletişim-Hesaplama Çakışması
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        gorevler = ["Compute Layer 3", "All-Gather Layer 2", "Compute Layer 2", "Reduce-Scatter Grad 3", "Compute Layer 1"]
        baslangic = [0, 0.2, 1.0, 1.0, 1.8]
        sure = [1.0, 0.7, 0.8, 0.7, 0.9]
        renk_list = ["#3b82f6", "#f59e0b", "#3b82f6", "#10b981", "#3b82f6"]

        for i, (g, s, d, c) in enumerate(zip(gorevler, baslangic, sure, renk_list)):
            ax4.barh(g, d, left=s, color=c, edgecolor="#ffffff", height=0.5, alpha=0.85)

        ax4.set_xlabel("Zaman (Normalize Adım)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Backward Prefetch: İletişim ve Hesaplama Çakışması", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.grid(axis="x", linestyle=":", alpha=0.4)

        # -------------------------------------------------------------
        # PANEL 5: Katman Parametre Sharding Dağılımı (Simülasyon)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        katman_isimleri = [f"Katman {i+1}" for i in range(len(layer_stats))]
        tam_paramlar = [s["toplam_parametre_numel"] for s in layer_stats]
        shard_paramlar = [s["shard_parametre_numel"] for s in layer_stats]

        x_l = np.arange(len(katman_isimleri))
        ax5.plot(x_l, tam_paramlar, marker="o", color="#ef4444", label="Tam Parametre Sayısı (DDP)")
        ax5.plot(x_l, shard_paramlar, marker="s", color="#38bdf8", label="Shard Parametre Sayısı (FSDP)")
        ax5.set_xticks(x_l)
        ax5.set_xticklabels(katman_isimleri, fontsize=9)
        ax5.set_ylabel("Eleman Sayısı (Numel)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Katman Bazlı Parametre Sharding ($1/N$ Dağılımı)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.legend(loc="upper left", fontsize=8)
        ax5.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 6: GÜN 182 Özet & Çıkarım Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "GÜN 182: FSDP VE ZeRO-3 MİMARİ KARNE\n"
            "----------------------------------------------------\n"
            "• Sharding Seviyesi   : FULL_SHARD (ZeRO-3)\n"
            "• Sharded Bileşenler  : Ağırlık + Gradyan + AdamW (m, v)\n"
            "• Bellek Bölüşümü     : O(1/N) Kusursuz Sıfır Artıklık\n"
            "• 70B Model 64 GPU    : 1,120 GB -> 17.5 GB (%98.4 Tasarruf!)\n"
            "• İletişim Primitifi  : All-Gather (Fwd/Bwd) + Reduce-Scatter\n"
            "• İletişim Hacmi      : Standart DDP'nin 1.5x katı (Kabul edilebilir)\n"
            "• Prefetching         : Backward All-Gather örtüşmesi\n"
            "----------------------------------------------------\n"
            "SONUÇ: 70B+ LLM'lerin 80GB VRAM kısıtını kıran, trilyon\n"
            "parametreli modelleri kümede eğiten modern endüstri standardı!"
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
