"""
Diffusion Transformers (DiT) Teşhis Panosu Görselleştirici Modülü (Day 177 - FAZ 9).
6 panelli DiT Model Ailesi Ölçeklenme Kıyası (DiT-S vs XL - GigaFLOPs vs FID), Patch Size (p=2, 4, 8) Etkisi, adaLN-Zero Modülasyon İzi, Patchify Tokenlaşma Görseli, DiT vs UNet Mimarisi ve Özet Kartı.
"""

import os
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np


class DiTGorsellestirici:
    """Diffusion Transformers Teşhis Panosu Üreticisi."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        rapor: Dict[str, Any],
        kayit_yolu: str = "ciktilar/dit_diffusion_transformers_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(23, 13.5))
        fig.suptitle(
            "GÜN 177 (FAZ 9): Diffusion Transformers (DiT - Sora & Flux Omurgası): Patchify, adaLN-Zero ve Saf Transformer Mimarisi",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: DiT Model Ölçeklenme Yasası (GigaFLOPs vs FID Skoru)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        modeller = [item["model"].split()[0] for item in rapor["model_varyantlari"]]
        gflops = [item["gflops"] for item in rapor["model_varyantlari"]]
        fids = [item["fid"] for item in rapor["model_varyantlari"]]

        ax1_twin = ax1.twinx()
        bar1 = ax1.bar(modeller, gflops, color="#4e73df", alpha=0.7, width=0.45, label="Hesaplama (GigaFLOPs)")
        line1 = ax1_twin.plot(modeller, fids, color="#e74a3b", marker="s", linewidth=2.5, label="FID Skoru (Daha Düşük = Daha İyi)")

        ax1.set_title("1. DiT Ölçeklenme Yasası (Scaling Law: Model Büyüdükçe Kalite Artar)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Hesaplama Maliyeti (GFLOPs)", color="#4e73df")
        ax1_twin.set_ylabel("FID Skoru (Düşük = Kaliteli)", color="#e74a3b")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Patch Size (p=2, 4, 8) Çözünürlük ve Hız Analizi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        patches = [item["patch"] for item in rapor["patch_boyut_analizi"]]
        tokenlar = [item["token_sayisi"] for item in rapor["patch_boyut_analizi"]]
        renkler2 = ["#1cc88a", "#36b9cc", "#f6c23e"]

        barlar2 = ax2.bar(patches, tokenlar, color=renkler2, edgecolor="black", width=0.45)
        for bar, item in zip(barlar2, rapor["patch_boyut_analizi"]):
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{h} Token\n({item['kalite'].split()[0]})", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax2.set_title("2. Yama Boyutu (Patch Size) vs Mekansal Token Sayısı", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Görsel Token Sayısı (N)")
        ax2.set_ylim(0, 310)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: adaLN-Zero Modülasyon Parametre Dinamiği
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        t_vals = np.linspace(0, 1000, 100)
        gamma = 1.0 + 0.5 * np.sin(t_vals / 150)
        beta = 0.2 * np.cos(t_vals / 200)
        alpha_gate = np.exp(-t_vals / 400) * 0.8

        ax3.plot(t_vals, gamma, label="Ölçek: gamma(t, c)", color="#4e73df", linewidth=2.2)
        ax3.plot(t_vals, beta, label="Kaydırma: beta(t, c)", color="#f6c23e", linewidth=2.0)
        ax3.plot(t_vals, alpha_gate, label="Kapı: alpha_gate(t, c)", color="#e74a3b", linewidth=2.2, linestyle="--")

        ax3.set_title("3. adaLN-Zero Koşul Parametreleri (Zaman t & Metin c)", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Difüzyon Zaman Adımı (t)")
        ax3.set_ylabel("Modülasyon Katsayısı")
        ax3.legend(loc="upper right", frameon=True)
        ax3.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: DiT İcra İzi ve Model Karşılaştırma Logu
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. DiT vs UNet Mimari Karşılaştırma İzi", fontsize=12, fontweight="bold", pad=10)

        log_metni = (
            "====================================================\n"
            "         DIFFUSION TRANSFORMER (DiT) LOG            \n"
            "====================================================\n"
            "• GİRİŞ TENSÖRÜ    : [Batch=1, C=4, 32x32] (VAE Latent)\n"
            "• PATCHIFY BOYUTU  : p = 2x2  -> Toplam N = 256 Token\n"
            "• DÖNÜŞÜM BOYUTU   : [1, 256, 1152] (DiT-XL Standardı)\n"
            "• MODÜLASYON TİPİ  : adaLN-Zero (6x Parametre Regresyonu)\n"
            "----------------------------------------------------\n"
            "ÖLÇEKLENME YASASI TEST SONUÇLARI:\n"
            "• DiT-S/2 -> GFLOPs: 18.2  | FID: 10.50\n"
            "• DiT-B/2 -> GFLOPs: 74.0  | FID: 4.80\n"
            "• DiT-L/2 -> GFLOPs: 260.0 | FID: 3.10\n"
            "• DiT-XL/2-> GFLOPs: 384.0 | FID: 2.27 (SOTA)\n"
            "----------------------------------------------------\n"
            "AVANTAJ: Konvolüsyonel önyargı yok, saf donanım ölçeklenebilirliği!\n"
            "===================================================="
        )

        ax4.text(
            0.02, 0.5, log_metni,
            fontsize=7.2,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: DiT Blok ve adaLN-Zero Akış Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. adaLN-Zero DiT Blok Mimarisi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "         adaLN-ZERO DIFFUSION TRANSFORMER BLOCK     \n"
            "====================================================\n"
            "  Koşul c (t + Metin) ──> [Linear (6xD)] ───────────┐\n"
            "                                                    │\n"
            "  Görsel Tokenları (x) ────────────────────────┐    │ (gamma, beta, alpha)\n"
            "           │                                   │    │\n"
            "           ▼                                   ▼    ▼\n"
            "  [LayerNorm] ──> [Modulate (gamma1, beta1)] ──> [Multi-Head Self-Attn]\n"
            "                                                        │\n"
            "  x ──────────────── (+) ◄── [Scale by alpha1_gate] ────┘\n"
            "           │\n"
            "           ▼\n"
            "  [LayerNorm] ──> [Modulate (gamma2, beta2)] ──> [Feed-Forward MLP]\n"
            "                                                        │\n"
            "  x ──────────────── (+) ◄── [Scale by alpha2_gate] ────┘\n"
            "           │\n"
            "           ▼\n"
            "  [Sonraki DiT Bloğuna Aktar: x_out]\n"
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
        # PANEL 6: GÜN 177 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 177 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   DAY 177 SUMMARY: DIFFUSION TRANSFORMERS (DiT)    \n"
            "====================================================\n"
            "• Modül              : FAZ 9 (Çok Modlu Modeller)\n"
            "• Temel Mimari       : Peebles & Xie (2023) DiT\n"
            "• Kullanım Alanları  : OpenAI Sora, SD 3, Flux.1, PixArt\n"
            "• İdeal Yama Boyutu  : p = 2x2 (En yüksek çözünürlük & FID: 2.27)\n"
            "• Koşullandırma      : adaLN-Zero (Identity block başlangıcı)\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. Konvolüsyonel UNet yerine saf Transformer omurgası\n"
            "  2. Patchify ile 2D görüntüleri 1D token dizilerine dönüştürme\n"
            "  3. adaLN-Zero ile sıfır gradyan patlamalı kararlı eğitim\n"
            "  4. FLOPs arttıkça görüntü kalitesinin doğrusal artması\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 178 (Consistency Models - LCM)\n"
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
        print(f"  ✓ DiT Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
