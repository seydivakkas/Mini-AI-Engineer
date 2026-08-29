"""
Latent Diffusion Modelleri Teşhis Panosu Görselleştirici Modülü (Day 172 - FAZ 9).
6 panelli İleri Difüzyon İzi (z_0 -> z_T), Geri Difüzyon İzi (z_T -> z_0), Alpha_bar Zaman Çizelgesi, Gürültü Kestirim MSE Dağılımı, LDM Mimarisi ve Özet Kartı.
"""

import os
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np


class LDMGorsellestirici:
    """Latent Diffusion Modeli Teşhis Panosu Üreticisi."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        difuzyon_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/latent_diffusion_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(23, 13.5))
        fig.suptitle(
            "GÜN 172 (FAZ 9): Latent Diffusion Modelleri (LDM / Stable Diffusion) — VAE Gizli Uzayı İleri/Geri Difüzyon Matematiği",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: İleri Difüzyon İzi (Forward Diffusion Process: z_0 -> z_T)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        adim_etiketleri = [f"t={s['t']}\n(%{int(s['gurultu_orani']*100)})" for s in difuzyon_raporu["adimlar"]]
        gurultu_seviyeleri = [s["gurultu_orani"] for s in difuzyon_raporu["adimlar"]]

        ax1.plot(adim_etiketleri, gurultu_seviyeleri, marker="o", color="#e74a3b", linewidth=2.5, markersize=8)
        for i, val in enumerate(gurultu_seviyeleri):
            ax1.text(i, val + 0.04, f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax1.set_title("1. İleri Difüzyon Gürültü Ekleme İzi (q(z_t | z_0))", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Eklenen Gauss Gürültüsü Oranı")
        ax1.set_ylim(-0.05, 1.15)
        ax1.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Gürültü Zaman Çizelgesi Kıyası: Linear vs Cosine alpha_bar_t
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        timesteps = np.linspace(0, 1000, 100)
        linear_alpha_bar = 1.0 - (timesteps / 1000.0) ** 1.5
        cosine_alpha_bar = np.cos((timesteps / 1000.0) * (np.pi / 2)) ** 2

        ax2.plot(timesteps, linear_alpha_bar, label="Linear Schedule", color="#4e73df", linewidth=2.2)
        ax2.plot(timesteps, cosine_alpha_bar, label="Cosine Schedule (Önerilen)", color="#1cc88a", linewidth=2.2, linestyle="--")

        ax2.set_title("2. Kümülatif Varyans Korunumu (alpha_bar_t)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Difüzyon Zaman Adımı (t)")
        ax2.set_ylabel("Kalan Orijinal Sinyal Gücü (alpha_bar)")
        ax2.legend(loc="upper right", frameon=True)
        ax2.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: VAE Gizli Uzayı vs Piksel Uzayı Hesaplama Kıyası
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        uzaylar = ["Piksel Uzayı\n(3 x 512 x 512)", "VAE Gizli Uzayı\n(4 x 64 x 64)"]
        bellek_mb = [786.432, 16.384]  # Tensör boyutları katsayısı
        renkler3 = ["#e74a3b", "#2e59d9"]

        barlar3 = ax3.bar(uzaylar, bellek_mb, color=renkler3, edgecolor="black", width=0.45)
        for bar in barlar3:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 15, f"{h:.1f}k", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax3.set_title("3. Hesaplama & Bellek Tasarrufu (64 Kat Kazanç)", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Tensör Eleman Sayısı (Bin)")
        ax3.set_ylim(0, 900)
        ax3.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Geri Difüzyon Örnekleme ve Gürültü Temizleme İzi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. Geri Difüzyon (Reverse Sampling) İcra İzi", fontsize=12, fontweight="bold", pad=10)

        ornekleme_metni = (
            "====================================================\n"
            "       REVERSE DIFFUSION SAMPLING LOG (z_T -> z_0)  \n"
            "====================================================\n"
            "BAŞLANGIÇ : z_T ~ N(0, I) (Saf Gauss Gürültüsü)\n"
            "----------------------------------------------------\n"
            "• t = 1000 ──> 750 : Kaba kompozisyon ve küresel şekiller belirdi\n"
            "• t = 750  ──> 500 : Ana hatlar ve semantik bölgeler ayrıştı\n"
            "• t = 500  ──> 250 : Doku ve nesne detayları netleşti\n"
            "• t = 250  ──> 0   : Yüksek frekanslı keskin detaylar tamamlandı\n"
            "----------------------------------------------------\n"
            "VAE DECODER : x_hat = D(z_0) [512 x 512 Piksel Görüntü]\n"
            f"GÜRÜLTÜ KESTİRİM MSE : {difuzyon_raporu['ortalama_gurultu_kestirim_mse']:.4f}\n"
            f"ÖRNEKLEME HIZI       : {difuzyon_raporu['ornekleme_hizi_fps']} FPS\n"
            "===================================================="
        )

        ax4.text(
            0.02, 0.5, ornekleme_metni,
            fontsize=7.2,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: LDM / Stable Diffusion Mimarisi Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Latent Diffusion Mimarisi Şeması", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "          LATENT DIFFUSION ARCHITECTURE (LDM)       \n"
            "====================================================\n"
            "  [Görüntü (512x512)] ──> [VAE Encoder E] ──> [z_0 (64x64x4)]\n"
            "                                    │               \n"
            "  [Zaman (t)] ──> [Sinusoidal Time] ├──> İleri Difüzyon q(z_t|z_0)\n"
            "                        │           │               \n"
            "                        ▼           ▼               \n"
            "                [Denoising UNet eps_theta(z_t, t)]  \n"
            "                        │                           \n"
            "                        ▼  (Kestirilen Gürültü eps) \n"
            "  [Geri Difüzyon Örnekleme p_theta(z_{t-1}|z_t)]    \n"
            "                        │                           \n"
            "                        ▼                           \n"
            "  [Temiz z_0] ──> [VAE Decoder D] ──> [Üretilen Görüntü]\n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, sema_metni,
            fontsize=7.2,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 172 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 172 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   DAY 172 SUMMARY: LATENT DIFFUSION MODELS (LDM)   \n"
            "====================================================\n"
            "• Modül              : FAZ 9 (Çok Modlu Modeller)\n"
            "• Temel Mimariler    : Stable Diffusion, SDXL, DDPM\n"
            "• Uzay Tipi          : VAE Gizli Uzayı (Latent Space 4x64x64)\n"
            f"• Hesaplama Kazancı  : {difuzyon_raporu['hesaplama_tasarrufu']}\n"
            f"• Gürültü Kestirim   : MSE = {difuzyon_raporu['ortalama_gurultu_kestirim_mse']:.4f}\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. Piksel uzayında devasa maliyetli difüzyondan kurtulma\n"
            "  2. Sinüzoidal zaman gömmesi ile UNet koşullandırma\n"
            "  3. Cosine vs Linear gürültü zaman çizelgeleri\n"
            "  4. DDPM ileri gürültü ekleme ve geri örnekleme matematiği\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 173 (Classifier-Free Guidance - CFG)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, ozet_metin,
            fontsize=7.6,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Latent Diffusion Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
