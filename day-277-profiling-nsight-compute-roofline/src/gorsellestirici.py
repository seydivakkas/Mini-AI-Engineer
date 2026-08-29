"""
Day 277 (FAZ 14): NVIDIA Nsight Compute & Roofline 6 Panelli Görselleştirici Modülü.
"""

from typing import Dict, Any
import os
import matplotlib.pyplot as plt
import numpy as np


class RooflineGorsellestirici:
    """FAZ 14 Hiyerarşik Roofline Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/nsight_roofline_paneli.png",
    ):
        """6 Panelli Nsight Roofline Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 277 (FAZ 14): NVIDIA NSIGHT COMPUTE & HİYERARŞİK ROOFLINE MODELİ — DONANIM DARBOĞAZI ANALİZİ",
            fontsize=15,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Hiyerarşik Roofline Eğrisi (HBM3, L2, SRAM vs Peak TFLOPS)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        intensities = profil_raporu["intensities"]
        ax1.loglog(intensities, profil_raporu["roof_hbm3"], "-", color="#ef4444", label="HBM3 (3.35 TB/s)", linewidth=2)
        ax1.loglog(intensities, profil_raporu["roof_l2"], "--", color="#f59e0b", label="L2 Cache (12 TB/s)", linewidth=2)
        ax1.loglog(intensities, profil_raporu["roof_sram"], "-.", color="#10b981", label="SRAM / Shared Mem (33 TB/s)", linewidth=2)
        
        # Ridge Point Dikey Çizgisi
        ax1.axvline(x=profil_raporu["ridge_point"], color="#38bdf8", linestyle=":", label=f"Ridge Point ({profil_raporu['ridge_point']:.1f} F/B)")

        # Kernel Noktaları
        kernel_renkleri = ["#ef4444", "#f97316", "#38bdf8", "#10b981"]
        for idx, k in enumerate(profil_raporu["kernel_analizleri"]):
            ax1.scatter(
                k["arithmetic_intensity_flop_per_byte"],
                k["achieved_tflops"],
                color=kernel_renkleri[idx],
                s=90,
                zorder=5,
                edgecolor="#ffffff",
                label=f"{k['kernel_name']} ({k['achieved_tflops']:.0f} TF)",
            )

        ax1.set_xlabel("Aritmetik Yoğunluk (FLOP / Byte)", fontsize=10, color="#cbd5e1")
        ax1.set_ylabel("Hesaplama Başarımı (TFLOPS)", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. NVIDIA H100 Hiyerarşik Roofline Modeli", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(True, which="both", linestyle=":", alpha=0.3)
        ax1.legend(loc="lower right", fontsize=7.5, facecolor="#1e293b", edgecolor="#38bdf8")

        # -------------------------------------------------------------
        # PANEL 2: LLM Çekirdekleri Aritmetik Yoğunluğu (FLOP/Byte)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        k_adlari = [k["kernel_name"] for k in profil_raporu["kernel_analizleri"]]
        k_yogunluk = [k["arithmetic_intensity_flop_per_byte"] for k in profil_raporu["kernel_analizleri"]]
        
        b2 = ax2.bar(k_adlari, k_yogunluk, color=kernel_renkleri, width=0.45)
        ax2.set_yscale("log")
        ax2.set_ylabel("Aritmetik Yoğunluk (FLOP / Byte)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Aritmetik Yoğunluk (I = FLOP / Byte - Log)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h * 1.2, f"{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9.5)

        # -------------------------------------------------------------
        # PANEL 3: Çekirdek Ulaşılan TFLOPS (Hız)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        k_tflops = [k["achieved_tflops"] for k in profil_raporu["kernel_analizleri"]]
        b3 = ax3.bar(k_adlari, k_tflops, color=kernel_renkleri, width=0.45)
        ax3.set_yscale("log")
        ax3.set_ylabel("Ulaşılan Hız (TFLOPS - Log)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Donanım Throughput (6.7 TF -> 1920 TF)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h * 1.2, f"{h:.1f} TF", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9.5)

        # -------------------------------------------------------------
        # PANEL 4: Nsight Warp Scheduler Stall Dağılımı (%)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        stalls = profil_raporu["warp_stalls"]["sebepler"]
        oranlar = profil_raporu["warp_stalls"]["oranlar_yuzde"]
        stall_renkler = ["#ef4444", "#f59e0b", "#3b82f6", "#10b981", "#8b5cf6"]

        b4 = ax4.bar(stalls, oranlar, color=stall_renkler, width=0.45)
        ax4.set_ylabel("Warp Stall Oranı (%)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Nsight Warp Stall Sebepleri (Memory Throttling)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(0, 70)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"%{h:.0f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9.5)

        # -------------------------------------------------------------
        # PANEL 5: Donanım Sınırına Göre MFU / Verimlilik Oranı (%)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        verimler = [k["hardware_efficiency_pct"] for k in profil_raporu["kernel_analizleri"]]
        b5 = ax5.bar(k_adlari, verimler, color="#38bdf8", width=0.45)
        ax5.set_ylabel("Ulaşılabilir Tepeye Oran (%)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Teorik Donanım Tavanına Ulaşma Verimi", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.set_ylim(0, 120)
        ax5.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b5:
            h = b.get_height()
            ax5.text(b.get_x() + b.get_width() / 2.0, h + 2.0, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9)

        # -------------------------------------------------------------
        # PANEL 6: Nsight & Roofline Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "NSIGHT COMPUTE & ROOFLINE TEŞHİS RAPORU\n"
            "====================================================\n"
            "• Hedef GPU          : NVIDIA H100 SXM5 (80GB HBM3)\n"
            "• HBM3 Bant Genişliği: 3.35 TB/s | L2: 12 TB/s | SRAM: 33 TB/s\n"
            "• Peak FP16 Tensor   : 1979 TFLOPS\n"
            "• Ridge Point Eşiği  : 590.7 FLOP / Byte\n"
            "• Softmax Durumu     : I=2.0 F/B (Aşırı Memory-Bound / %52 Stall)\n"
            "• FlashAttention-2   : I=160.0 F/B (SRAM Fused | 80x Hızlanma)\n"
            "• Fused FP8 GEMM     : I=851.0 F/B (Compute-Bound | 1920 TFLOPS)\n"
            "• Birincil Çözüm     : Memory-Bound -> SRAM Tile Füzyonu\n"
            "----------------------------------------------------\n"
            "FAZ 14 GÜN 277 NSIGHT ROOFLINE MODÜLÜ TAMAMLANDI!\n"
            "Sırada: Day 278 (AMD ROCm & HIP Taşınabilirliği)"
        )

        ax6.text(
            0.05,
            0.5,
            ozet_metin,
            fontsize=9.2,
            family="monospace",
            color="#f8fafc",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.9),
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close()
