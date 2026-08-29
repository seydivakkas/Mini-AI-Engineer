"""
Token Sıkıştırma Teşhis Panosu Görselleştirici Modülü (Day 162 - FAZ 9).
6 panelli Token Sayısı, Sıkıştırma Oranı, Attention Bellek Tasarrufu, Q-Former Dikkat Haritası, Mimari Kıyaslama ve Özet Kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class TokenSikistirmaGorsellestirici:
    """Token Sıkıştırma Teşhis Panosu Üreticisi."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        karsilastirma: Dict[str, Any],
        kayit_yolu: str = "ciktilar/vision_token_compression_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(23, 13.5))
        fig.suptitle(
            "GÜN 162 (FAZ 9): Görüntü Token Sıkıştırma — Q-Former, C-Abstractor ve Spatial Pooling Analizi",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        modeller = list(karsilastirma.keys())
        kisa_isimler = ["Ham ViT", "Spatial Pool", "C-Abstractor", "BLIP-2 Q-Former"]

        # -------------------------------------------------------------
        # PANEL 1: LLM'e Giren Görsel Token Sayısı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        token_sayilari = [karsilastirma[m]["token_sayisi"] for m in modeller]
        renkler1 = ["#e74a3b", "#f6c23e", "#36b9cc", "#1cc88a"]

        barlar1 = ax1.bar(kisa_isimler, token_sayilari, color=renkler1, edgecolor="black", width=0.45)
        for bar in barlar1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 3.0, f"{int(h)} Token", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax1.set_title("1. Görsel Token Sayısı (LLM Girdi Boyutu)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Token Sayısı (Adet)")
        ax1.set_ylim(0, 300)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Doğrudan Token Sıkıştırma Oranı (%)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        oranlar = [karsilastirma[m]["sikistirma_orani"] for m in modeller]

        barlar2 = ax2.bar(kisa_isimler, oranlar, color=renkler1, edgecolor="black", width=0.45)
        for bar in barlar2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax2.set_title("2. Token Sıkıştırma Oranı (%)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Sıkıştırma Yüzdesi (%)")
        ax2.set_ylim(0, 105)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: LLM Self-Attention O(N^2) Bellek Tasarrufu
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        tasarruf = [karsilastirma[m]["attention_bellek_tasarrufu"] for m in modeller]

        barlar3 = ax3.bar(kisa_isimler, tasarruf, color=["#6c757d", "#4e73df", "#36b9cc", "#1cc88a"], edgecolor="black", width=0.45)
        for bar in barlar3:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax3.set_title("3. Self-Attention O(N²) Bellek & FLOPs Tasarrufu", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Tasarruf Oranı (%)")
        ax3.set_ylim(0, 110)
        ax3.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Temsili Q-Former Cross-Attention Isı Haritası
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        np.random.seed(42)
        # 8 Query token x 16 Görsel Patch (Özet görünüm)
        attn_matrix = np.random.uniform(0.1, 0.9, (8, 16))
        attn_matrix = attn_matrix / attn_matrix.sum(axis=1, keepdims=True)

        im = ax4.imshow(attn_matrix, cmap="viridis", aspect="auto")
        ax4.set_title("4. Q-Former Query x Visual Cross-Attention Matrisi", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Görsel Patch Tokenları (Örnek 16 Patch)")
        ax4.set_ylabel("Öğrenilebilir Query Token (1-8)")
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

        # -------------------------------------------------------------
        # PANEL 5: 3 Sıkıştırma Yönteminin Mimari Kıyası
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. 3 Sıkıştırma Mimarisinin Çalışma Prensibi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "         VISION TOKEN COMPRESSION TAXONOMY          \n"
            "====================================================\n"
            "1. SPATIAL POOLING (2x2 / 4x4):                     \n"
            "   [256 Patch] -> [2D Grid (16x16)] -> [Pool (8x8)] \n"
            "   -> 64 Token (Parametresiz, Hızlı, Düşük Maliyet)\n"
            "----------------------------------------------------\n"
            "2. C-ABSTRACTOR (Convolutional):                    \n"
            "   [256 Patch] -> [Depthwise Conv 3x3] -> [8x8 Grid]\n"
            "   -> 64 Token (Yerel Doku & Kenar Bilgisini Korur) \n"
            "----------------------------------------------------\n"
            "3. BLIP-2 Q-FORMER (Cross-Attention):               \n"
            "   [256 Patch (K,V)] + [32 Query Tokens (Q)]        \n"
            "   -> 32 Token (%87.5 Sıkıştırma, Anlamsal Damıtma) \n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, sema_metni,
            fontsize=7.3,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 162 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 162 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   DAY 162 SUMMARY: VISION TOKEN COMPRESSION        \n"
            "====================================================\n"
            "• Modül              : FAZ 9 (Çok Modlu Modeller)\n"
            "• Girdi Token Sayısı : 256 Token (ViT-14x14)\n"
            "• Zirve Sıkıştırma   : BLIP-2 Q-Former (32 Token)\n"
            "• Attention Bellek   : %98.4 Bellek Tasarrufu!\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. LLM dikkat matrisini O(256^2) yerine O(32^2)'ye düşürme\n"
            "  2. Çapraz dikkat ile önemli anlamsal pikselleri çekme\n"
            "  3. Konvolüsyonel Abstractor ile yerel uzamsal füzyon\n"
            "  4. Yüksek çözünürlüklü VLM'lerde bağlam patlamasını önleme\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 163 (Visual Instruction Tuning - SFT)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, ozet_metin,
            fontsize=7.8,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Token Sıkıştırma Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
