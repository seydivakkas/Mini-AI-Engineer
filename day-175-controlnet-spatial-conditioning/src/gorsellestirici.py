"""
ControlNet Teşhis Panosu Görselleştirici Modülü (Day 175 - FAZ 9).
6 panelli Mekansal Koşul Uyumu (Canny, Depth, Pose), Zero-Conv Sıfır Başlangıç Analizi, Kontrol Gücü Eğrisi, Koşullu İcra İzi, ControlNet Mimarisi ve Özet Kartı.
"""

import os
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np


class ControlNetGorsellestirici:
    """ControlNet Mekansal Koşullandırma Teşhis Panosu Üreticisi."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        rapor: Dict[str, Any],
        kayit_yolu: str = "ciktilar/controlnet_spatial_conditioning_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(23, 13.5))
        fig.suptitle(
            "GÜN 175 (FAZ 9): ControlNet — Canny Kenar, Derinlik ve OpenPose ile Mekansal Koşullu Görüntü Üretimi (Zero-Convolution)",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Mekansal Koşul Uyumu Skoru (Canny vs Depth vs Pose)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        tipler = [item["tip"].split(" (")[0] for item in rapor["kosul_tipleri"]]
        skorlar = [item["uyum_skoru"] * 100 for item in rapor["kosul_tipleri"]]
        renkler1 = ["#4e73df", "#1cc88a", "#f6c23e"]

        barlar1 = ax1.bar(tipler, skorlar, color=renkler1, edgecolor="black", width=0.5)
        for bar in barlar1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"%{h:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax1.set_title("1. Mekansal Koşul Sadakat Skoru (%)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Mekansal Uyum (Spatial Alignment %)")
        ax1.set_ylim(80, 102)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Zero-Convolution Ağırlık Büyümesi ve Gradyan Akışı
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        egitim_adimlari = np.linspace(0, 10000, 100)
        zero_conv_weight = 1.0 - np.exp(-egitim_adimlari / 2500)

        ax2.plot(egitim_adimlari, zero_conv_weight, color="#e74a3b", linewidth=2.5, label="Zero-Conv Ağırlık Büyümesi")
        ax2.scatter([0], [0], color="black", s=80, zorder=5, label="Başlangıç: W=0, b=0 (Sıfır Etki)")

        ax2.set_title("2. Zero-Conv Ağırlık Öğrenme Eğrisi (Zararsız Başlangıç)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Eğitim Adımı (Training Steps)")
        ax2.set_ylabel("Sıfır-Konvolüsyon Ağırlık Normu")
        ax2.legend(loc="lower right", frameon=True)
        ax2.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Control Weight (0.0 - 2.0) vs Prompt Etkisi Dengesi
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        weights = np.linspace(0.0, 2.0, 50)
        spatial_control = 1.0 / (1.0 + np.exp(-4 * (weights - 0.7)))
        prompt_freedom = 1.0 - (weights / 2.0) ** 1.5

        ax3.plot(weights, spatial_control, label="Mekansal Şekil Sadakati", color="#4e73df", linewidth=2.2)
        ax3.plot(weights, prompt_freedom, label="İstem Yaratıcılığı (Prompt Freedom)", color="#f6c23e", linewidth=2.2, linestyle="--")
        ax3.axvline(x=1.0, color="#1cc88a", linestyle=":", linewidth=2, label="İdeal Denge (Weight=1.0)")

        ax3.set_title("3. Control Weight Denge Analizi", fontsize=12, fontweight="bold")
        ax3.set_xlabel("ControlNet Ağırlık Katsayısı")
        ax3.set_ylabel("Etki Gücü [0 - 1.0]")
        ax3.legend(loc="center right", frameon=True)
        ax3.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Mekansal Koşullu İcra İzi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. ControlNet Koşullu İcra İzi", fontsize=12, fontweight="bold", pad=10)

        icra_metni = (
            "====================================================\n"
            "       CONTROLNET SPATIAL CONDITIONING LOG          \n"
            "====================================================\n"
            "GİRİŞ İPUÇLARI (SPATIAL HINTS):\n"
            "• Canny Edge  : Kenar Çizgileri [3 x 512 x 512]\n"
            "• Depth Map   : 3D Derinlik [3 x 512 x 512]\n"
            "• OpenPose    : 18-Eklem İskelet Pozu [3 x 512 x 512]\n"
            "----------------------------------------------------\n"
            "ZERO-CONV DURUMU:\n"
            "  - Adım 0  : out = 0.0 (Dondurulmuş model %100 saf)\n"
            "  - Adım 5k : Mekansal ipucu UNet decoder'a akmaya başladı\n"
            "----------------------------------------------------\n"
            f"ORTALAMA MEKANSAL UYUM : %{rapor['ortalama_mekansal_uyum']*100:.1f}\n"
            f"EĞİTİM KARARLILIĞI     : {rapor['zero_conv_egitim_kararliligi']}\n"
            "===================================================="
        )

        ax4.text(
            0.02, 0.5, icra_metni,
            fontsize=7.2,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: ControlNet Mimari Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Locked Model & Trainable Copy Şeması", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "          CONTROLNET LOCKED/TRAINABLE COPY          \n"
            "====================================================\n"
            "  [Dondurulmuş UNet Encoder] ───────┐ (Orijinal Ağırlık)\n"
            "           │ (Klonlandı)            │               \n"
            "           ▼                        │               \n"
            "  [Eğitilebilir Klon Encoder]        │               \n"
            "      ├── [Koşul Hint: Canny/Pose]  │               \n"
            "      └── [Giriş Zero-Conv (W=0)]   │               \n"
            "           │                        │               \n"
            "           ▼ (Çıkış Zero-Convolutions)             \n"
            "  [Control Residuals] ──────────────┴──> [Dondurulmuş UNet Decoder]\n"
            "                                                │   \n"
            "                                                ▼   \n"
            "  [Mekansal Olarak %100 Hizalı Kusursuz Üretim] ────┘\n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, sema_metni,
            fontsize=7.1,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 175 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 175 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   DAY 175 SUMMARY: CONTROLNET SPATIAL DIFFUSION    \n"
            "====================================================\n"
            "• Modül              : FAZ 9 (Çok Modlu Modeller)\n"
            "• Temel Buluş        : Zero-Convolution & Trainable Encoder Copy\n"
            "• Desteklenen İpuçları: Canny Edge, MiDaS Depth, OpenPose\n"
            f"• Mekansal Uyum      : %{rapor['ortalama_mekansal_uyum']*100:.1f}\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. Dondurulmuş modeli bozmadan piksel düzeyinde mekansal kontrol\n"
            "  2. Zero-Convolution ile eğitimin başında sıfır gürültü garantisi\n"
            "  3. Karakter pozlarını, oda perspektifini ve mimari hatları sabitleme\n"
            "  4. Oyun tasarımı, animasyon ve profesyonel CGI prodüksiyonu\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 176 (LoRA & DreamBooth Diffusion)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, ozet_metin,
            fontsize=7.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ ControlNet Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
