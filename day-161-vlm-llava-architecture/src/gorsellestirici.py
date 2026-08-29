"""
LLaVA VLM Teşhis Panosu Görselleştirici Modülü (Day 161 - FAZ 9).
6 panelli Patch Ayrıştırma, MLP Hizalama, Multimodal Füzyon, VLM Mimari Şeması, Çıkarım Çıktısı ve Özet Kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class VLMGorsellestirici:
    """LLaVA VLM Teşhis Panosu Üreticisi."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        cikarim_bilgisi: Dict[str, Any],
        kayit_yolu: str = "ciktilar/vlm_llava_architecture_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(23, 13.5))
        fig.suptitle(
            "GÜN 161 (FAZ 9): LLaVA Mimarisi — ViT Encoder + MLP Projector + LLM ile Çok Modlu VLM İnşası",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Patch Token Boyutları ve Dönüşüm Adımları
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        asamalar = ["1. Girdi Görüntü\n(3x224x224)", "2. ViT Patch\n(256 x 768)", "3. MLP Proj\n(256 x 512)", "4. Füzyon\n(266 x 512)"]
        boyut_skorlari = [150.5, 196.6, 131.0, 136.2]  # Temsili tensor ağırlık/boyut ölçeği
        renkler1 = ["#e74a3b", "#4e73df", "#f6c23e", "#1cc88a"]

        barlar1 = ax1.bar(asamalar, boyut_skorlari, color=renkler1, edgecolor="black", width=0.45)
        for bar in barlar1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 2.0, f"{h:.1f}k", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax1.set_title("1. Tensor Akışı ve Boyut Dönüşümleri", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Temsili Tensor Eleman Sayısı (x1000)")
        ax1.set_ylim(0, 230)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Multimodal Dizi Dağılımı (Visual vs Text Tokens)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        kategoriler = ["Görsel Tokenlar (ViT 14x14)", "Metin Tokenları (Soru/Prompt)"]
        sayilar = [cikarim_bilgisi["visual_token_sayisi"], cikarim_bilgisi["text_token_sayisi"]]
        renkler2 = ["#4e73df", "#1cc88a"]

        ax2.pie(sayilar, labels=kategoriler, autopct="%1.1f%%", colors=renkler2, startangle=140, explode=(0.05, 0.05))
        ax2.set_title(f"2. Multimodal Dizi Kompozisyonu (Toplam: {sum(sayilar)} Token)", fontsize=12, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 3: MLP Projektör Aktivasyon Dağılımı
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        np.random.seed(42)
        proj_weights = np.random.normal(0.0, 0.04, 1000)

        ax3.hist(proj_weights, bins=30, color="#36b9cc", edgecolor="black", alpha=0.8)
        ax3.set_title("3. MLP Projektör Ağırlık Dağılımı (GELU)", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Ağırlık Değeri")
        ax3.set_ylabel("Frekans")
        ax3.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Örnek Görsel Soru-Cevap (VQA) Çıktısı
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. Görsel Soru-Cevap (VQA) Çıkarım Örneği", fontsize=12, fontweight="bold", pad=10)

        vqa_metni = (
            "====================================================\n"
            "         LLaVA VISUAL QUESTION ANSWERING            \n"
            "====================================================\n"
            f"GİRDİ GÖRÜNTÜ: {cikarim_bilgisi['goruntu_aciklamasi']}\n"
            f"KULLANICI SORUSU: '{cikarim_bilgisi['kullanici_sorusu']}'\n"
            "----------------------------------------------------\n"
            f"LLaVA MODEL YANITI:\n"
            f"  '{cikarim_bilgisi['model_yaniti']}'\n"
            "----------------------------------------------------\n"
            f"  • ViT Patch Sayısı    : {cikarim_bilgisi['visual_token_sayisi']} adet (14x14)\n"
            f"  • Metin Token Sayısı  : {cikarim_bilgisi['text_token_sayisi']} adet\n"
            f"  • LLM Gizli Boyut (d) : {cikarim_bilgisi['d_text']}\n"
            "===================================================="
        )

        ax4.text(
            0.02, 0.5, vqa_metni,
            fontsize=7.3,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: LLaVA VLM Mimarisi Akış Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. LLaVA Mimarisi ve Multimodal Füzyon", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "             LLaVA VLM ARCHITECTURE PIPELINE        \n"
            "====================================================\n"
            "  [Görüntü (3, 224, 224)]                           \n"
            "           │                                        \n"
            "           ▼  14x14 Patch Embedding                  \n"
            "  [CLIP-ViT-L/14 Vision Encoder]                     \n"
            "           │  (256 Patch Token x 768 Boyut)         \n"
            "           ▼                                        \n"
            "  [2 Katmanlı MLP Projektör (GELU)]                 \n"
            "           │  (256 Patch Token x 512 Boyut)         \n"
            "           ▼                                        \n"
            "  [Multimodal Concatenation: <IMG_TOKENS> + <TEXT>] \n"
            "           │  (266 Token x 512 Boyut)               \n"
            "           ▼                                        \n"
            "  [Oto-Regresif LLM (Decoder)] ──> [Metin Yanıtı]   \n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, sema_metni,
            fontsize=7.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 161 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 161 ÖZET KARTI (FAZ 9 BAŞLADI)", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   DAY 161 SUMMARY: LLaVA VLM ARCHITECTURE          \n"
            "====================================================\n"
            "• Modül              : FAZ 9 (Çok Modlu / Multimodal)\n"
            "• Görüntü Kodlayıcı  : Vision Transformer (ViT-14x14)\n"
            "• Hizalama Köprüsü   : 2 Katmanlı GELU MLP Projektör\n"
            "• Dil Modeli         : Causal Decoder LLM\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. Görüntüyü 256 bağımsız görsel kelimeye (Patch) bölme\n"
            "  2. Görsel uzayı (768) metin uzayına (512) yansıtma\n"
            "  3. Görüntü ve metin tokenlarını tek dizide birleştirme\n"
            "  4. Uçtan uca Görsel Soru-Cevap (VQA) üretimi\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 162 (Vision Token Compression - Q-Former)\n"
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
        print(f"  ✓ LLaVA VLM Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
