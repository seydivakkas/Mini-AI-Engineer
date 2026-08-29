"""
Yeni Nesil FP4 / FP6 (Microscaling MXFP4 E2M1) 6 Panelli Görselleştirici Modülü (FAZ 14) (Day 264).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class MXFP4Gorsellestirici:
    """FAZ 14 Microscaling MXFP4 Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/mxfp4_microscaling_paneli.png",
    ):
        """6 Panelli MXFP4 Microscaling Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 264 (FAZ 14): YENİ NESİL FP4 / FP6 (MICROSCALING MXFP4 E2M1) KUANTİZASYON VE ÇEKİRDEK SİMÜLASYONU",
            fontsize=15,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        formatlar = ["1. FP16 Baseline\n(16-Bit Standart)", "2. FP8 E4M3\n(Hopper H100)", "3. INT4 PTQ\n(Klasik Kırpma)", "4. OCP MXFP4\n(Blackwell SOTA)"]
        renkler = ["#ef4444", "#f59e0b", "#a855f7", "#10b981"]

        # -------------------------------------------------------------
        # PANEL 1: OCP MXFP4 E2M1 Izgara Noktaları Dağılımı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        grid = np.array([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        ax1.stem(grid, np.ones_like(grid), linefmt="#38bdf8", markerfmt="C0o", basefmt="gray")
        ax1.set_xlabel("Temsil Edilen Sayı Değeri", fontsize=10, color="#cbd5e1")
        ax1.set_yticks([])
        ax1.set_title("1. FP4 E2M1 (1-Sign, 2-Exp, 1-Mantissa) Ayrık Izgara", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        for val in [-6.0, -4.0, -1.0, 0.0, 1.0, 4.0, 6.0]:
            ax1.text(val, 1.05, f"{val:g}", ha="center", va="bottom", color="#ffffff", fontsize=8.5, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 2: VRAM Bellek Tüketimi (%)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        bellek = [
            karsilastirma["bellek_tuketimi_yuzde"]["FP16_Baseline"],
            karsilastirma["bellek_tuketimi_yuzde"]["FP8_E4M3"],
            karsilastirma["bellek_tuketimi_yuzde"]["INT4_PTQ"],
            karsilastirma["bellek_tuketimi_yuzde"]["OCP_MXFP4_E2M1"],
        ]
        b2 = ax2.bar(formatlar, bellek, color=renkler, width=0.45)
        ax2.set_ylabel("VRAM Bellek Tüketimi (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Bellek Tüketimi (140GB -> 35GB | 4x Tasarruf)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(0, 115)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Sinyal-Gürültü Doğruluğu (SNR dB - Yüksek İyi)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        snr = [
            karsilastirma["sinyal_dogrulugu_snr_db"]["FP16_Baseline"],
            karsilastirma["sinyal_dogrulugu_snr_db"]["FP8_E4M3"],
            karsilastirma["sinyal_dogrulugu_snr_db"]["INT4_PTQ"],
            karsilastirma["sinyal_dogrulugu_snr_db"]["OCP_MXFP4_E2M1"],
        ]
        b3 = ax3.bar(formatlar, snr, color=renkler, width=0.45)
        ax3.set_ylabel("Sinyal Doğruluğu (SNR dB)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Sinyal Kalitesi (INT4: 22dB -> MXFP4: 39.5dB)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.set_ylim(0, 55)
        ax3.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"{h:.1f} dB", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 4: Blackwell B200 Hesaplama Gücü (PFLOPS)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        pflops = [
            karsilastirma["donanim_pflops_b200"]["FP16_Baseline"],
            karsilastirma["donanim_pflops_b200"]["FP8_E4M3"],
            karsilastirma["donanim_pflops_b200"]["INT4_PTQ"],
            karsilastirma["donanim_pflops_b200"]["OCP_MXFP4_E2M1"],
        ]
        b4 = ax4.bar(formatlar, pflops, color=renkler, width=0.45)
        ax4.set_ylabel("Hesaplama Gücü (PFLOPS)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Blackwell B200 Gücü (5.0 -> 20.0 PFLOPS | 4x)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(0, 24)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 0.5, f"{h:.1f} PF", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 5: Token Çıkarım Gecikmesi (ms - Düşük İyi)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        gecikme = [
            karsilastirma["cikarim_gecikmesi_ms"]["FP16_Baseline"],
            karsilastirma["cikarim_gecikmesi_ms"]["FP8_E4M3"],
            karsilastirma["cikarim_gecikmesi_ms"]["INT4_PTQ"],
            karsilastirma["cikarim_gecikmesi_ms"]["OCP_MXFP4_E2M1"],
        ]
        b5 = ax5.bar(formatlar, gecikme, color=renkler, width=0.45)
        ax5.set_ylabel("Gecikme (ms/token - Düşük İyi)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Çıkarım Gecikmesi (18.5ms -> 4.6ms | 4x Hızlı)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.set_ylim(0, 22)
        ax5.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b5:
            h = b.get_height()
            ax5.text(b.get_x() + b.get_width() / 2.0, h + 0.5, f"{h:.1f} ms", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 6: OCP MXFP4 Performans ve Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "OCP MXFP4 MICROSCALING RAPORU\n"
            "====================================================\n"
            "• Standart Format     : OCP Microscaling MXFP4 E2M1\n"
            "• Blok Ölçekleme      : 32 Eleman Paylaşımlı Üs (E8M0)\n"
            "• Dinamik Aralık      : [-6.0, +6.0] Logaritmik Dağılım\n"
            "• VRAM Bellek Kazancı : 4x Kat Tasarruf (140GB -> 35GB)\n"
            "• Sinyal Kalitesi     : 39.5 dB SNR (Kırpma Hatasız)\n"
            "• B200 Zirve Gücü     : 20.0 PFLOPS (4x Artış)\n"
            "• Çıkarım Gecikmesi   : 4.6 ms/token (4x Hızlanma)\n"
            "----------------------------------------------------\n"
            "FAZ 14 FP4 KUANTİZASYON MODÜLÜ TAMAMLANDI!\n"
            "Sırada: Day 265 (Triton Fused MoE Expert Routing)"
        )

        ax6.text(
            0.05,
            0.5,
            ozet_metin,
            fontsize=9.5,
            family="monospace",
            color="#f8fafc",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.9),
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close()
