"""
DeepSpeed ZeRO-Offload & ZeRO-Infinity 6 Panelli Görselleştirici Modülü (Day 183 - FAZ 10).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class ZeROGorsellestirici:
    """ZeRO-Offload ve ZeRO-Infinity 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def zero_offload_paneli_olustur(
        cls,
        profil_raporu: List[Dict[str, Any]],
        kayit_yolu: str = "ciktilar/deepspeed_zero123_offload_paneli.png",
    ):
        """6 Panelli ZeRO-Offload ve ZeRO-Infinity Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 183: DEEPSPEED ZeRO-OFFLOAD & ZeRO-INFINITY BELLEK VE PCIe ANALİZİ",
            fontsize=18,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Bellek Katmanlaşması (GPU vs CPU RAM vs NVMe SSD)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        mimariler = ["DDP", "FSDP\n(8 GPU)", "ZeRO-Offload\n(1 GPU)", "ZeRO-Infinity\n(1 GPU)"]
        gpu_vram = [1043.1, 130.4, 260.8, 13.0]      # 70B Model GB
        cpu_ram = [0.0, 0.0, 782.3, 52.2]
        nvme_ssd = [0.0, 0.0, 0.0, 912.7]

        x_pos = np.arange(len(mimariler))
        ax1.bar(x_pos, gpu_vram, label="GPU VRAM (Pahalı/Hızlı)", color="#ef4444", width=0.5)
        ax1.bar(x_pos, cpu_ram, bottom=gpu_vram, label="Host CPU RAM (Geniş/Orta)", color="#3b82f6", width=0.5)
        ax1.bar(x_pos, nvme_ssd, bottom=np.array(gpu_vram) + np.array(cpu_ram), label="NVMe SSD (Devasa/Ekonomik)", color="#10b981", width=0.5)

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(mimariler, fontsize=9)
        ax1.set_ylabel("Bellek Kullanımı (GB)", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. 70B Model İçin Hiyerarşik Bellek Dağılımı (GB)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.axhline(80, color="#f59e0b", linestyle="--", label="80GB A100 VRAM Limiti")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(axis="y", linestyle=":", alpha=0.4)

        # -------------------------------------------------------------
        # PANEL 2: ZeRO-Offload İleri/Geri & PCIe DMA Döngüsü
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        dongu_adimlari = ["1. Forward (GPU)", "2. Backward (GPU)", "3. D2H Grad Offload", "4. CPU AdamW Step", "5. H2D Param Update"]
        sure_oranlari = [30, 45, 10, 10, 5]
        renkler = ["#0284c7", "#0284c7", "#f59e0b", "#10b981", "#38bdf8"]

        ax2.pie(
            sure_oranlari,
            labels=dongu_adimlari,
            colors=renkler,
            autopct="%1.0f%%",
            startangle=140,
            textprops={"fontsize": 9, "color": "#f8fafc"},
            wedgeprops={"edgecolor": "#1e293b", "linewidth": 1.5},
        )
        ax2.set_title("2. ZeRO-Offload İterasyon Zaman Dağılımı (Compute vs DMA)", fontsize=11, color="#38bdf8", fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 3: Model Parametre Boyutu vs VRAM / CPU Belleği
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        modeller = [m["model_adi"] for m in profil_raporu]
        gpu_vals = [m["zero_offload_gpu_gb"] for m in profil_raporu]
        cpu_vals = [m["zero_offload_cpu_gb"] for m in profil_raporu]

        x = np.arange(len(modeller))
        width = 0.35
        ax3.bar(x - width / 2, gpu_vals, width, label="ZeRO-Offload GPU VRAM", color="#ef4444")
        ax3.bar(x + width / 2, cpu_vals, width, label="Host CPU RAM (AdamW)", color="#3b82f6")

        ax3.set_yscale("log")
        ax3.set_xticks(x)
        ax3.set_xticklabels(modeller, fontsize=8, rotation=15)
        ax3.set_ylabel("Bellek GB (Log Ölçek)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Model Boyutlarına Göre GPU VRAM vs CPU RAM", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.axhline(80, color="#f59e0b", linestyle=":", label="80 GB GPU Limiti")
        ax3.legend(loc="upper left", fontsize=8)
        ax3.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 4: PCIe Gen4/Gen5 Çift Tamponlama (Double Buffering Overlap)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        akiskatmanlar = ["GPU Compute Layer L", "PCIe H2D Layer L+1", "PCIe D2H Grad L-1", "GPU Compute Layer L+1"]
        b_zaman = [0.0, 0.1, 0.1, 1.0]
        s_zaman = [1.0, 0.8, 0.7, 1.0]
        c_list = ["#3b82f6", "#f59e0b", "#10b981", "#3b82f6"]

        for i, (ak, bz, sz, cl) in enumerate(zip(akiskatmanlar, b_zaman, s_zaman, c_list)):
            ax4.barh(ak, sz, left=bz, color=cl, edgecolor="#ffffff", height=0.45, alpha=0.85)

        ax4.set_xlabel("Zaman (Normalize Adım)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Double Buffering: PCIe DMA ile GPU Hesaplama Örtüşmesi", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.grid(axis="x", linestyle=":", alpha=0.4)

        # -------------------------------------------------------------
        # PANEL 5: CPU AdamW vs GPU AdamW Bellek ve Hesaplama İzi
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        kategoriler = ["Master W (FP32)", "Momentum (m)", "Variance (v)", "Param (FP16)", "Grad (FP16)"]
        gpu_statik = [4, 4, 4, 2, 2]       # Bayt/param
        offload_statik = [0, 0, 0, 2, 2]   # Bayt/param (GPU'da kalan)

        x_k = np.arange(len(kategoriler))
        w = 0.35
        ax5.bar(x_k - w / 2, gpu_statik, w, label="Standart DDP (Tümü GPU)", color="#ef4444")
        ax5.bar(x_k + w / 2, offload_statik, w, label="ZeRO-Offload (GPU Yükü)", color="#10b981")

        ax5.set_xticks(x_k)
        ax5.set_xticklabels(kategoriler, fontsize=8, rotation=15)
        ax5.set_ylabel("Parametre Başına Bellek (Bayt)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Optimizer Durumlarının CPU'ya Boşaltılması (16B -> 4B)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.legend(loc="upper right", fontsize=8)
        ax5.grid(axis="y", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 6: GÜN 183 Özet & Mimari Karar Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "GÜN 183: ZeRO-OFFLOAD & ZeRO-INFINITY MİMARİ RAPOR\n"
            "----------------------------------------------------\n"
            "• Hedef Cihazlar      : Host CPU RAM (DDR4/5) + NVMe SSD\n"
            "• Offload Bileşenleri : AdamW (m, v, master) + Katman Ağırlığı\n"
            "• VRAM Tasarrufu      : %75 (ZeRO-Offload), %98 (ZeRO-Infinity)\n"
            "• 70B Model Tek GPU   : 1,043 GB GPU -> 260 GB GPU + 782 GB CPU\n"
            "• 1T Model Eğitimi    : ZeRO-Infinity ile NVMe SSD üzerinden mümkün!\n"
            "• İletişim Örtüşmesi  : PCIe DMA Double-Buffering (Compute-Bound)\n"
            "• CPU AdamW           : AVX-512 Vektörel Hızlandırma\n"
            "----------------------------------------------------\n"
            "SONUÇ: Bütçe dostu tekli/küçük GPU sunucularında devasa\n"
            "modelleri çalıştırmanın en güçlü bellek mühendisliği anahtarı!"
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
