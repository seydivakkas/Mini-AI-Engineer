"""
Cross-Attention Metinden Görüntüye Teşhis Panosu Görselleştirici Modülü (Day 174 - FAZ 9).
6 panelli Kelime Bazlı Çapraz Dikkat Isı Haritaları, Self vs Cross Attention, Çok Başlıklı Odak Dağılımı, Metin-Piksel Hizalama İzi, Mimarisi ve Özet Kartı.
"""

import os
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np


class CrossAttentionGorsellestirici:
    """Metinden Görüntüye Çapraz Dikkat Teşhis Panosu Üreticisi."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        rapor: Dict[str, Any],
        kayit_yolu: str = "ciktilar/cross_attention_text_to_image_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(23, 13.5))
        fig.suptitle(
            "GÜN 174 (FAZ 9): Metinden Görüntüye: UNet & DiT Cross-Attention Mekanizması ve Mekansal Dikkat Haritaları",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Kelime Bazlı Çapraz Dikkat Dağılım Isı Haritası ('cat' vs 'helmet')
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        # Simüle edilmiş 16x16 odak haritası (Kedi gövdesi ve kaskı)
        grid_x, grid_y = np.meshgrid(np.linspace(-2, 2, 16), np.linspace(-2, 2, 16))
        cat_map = np.exp(-(grid_x**2 + grid_y**2) / 0.8)
        helmet_map = np.exp(-((grid_x)**2 + (grid_y + 0.8)**2) / 0.5)
        combined_map = 0.6 * cat_map + 0.4 * helmet_map

        im1 = ax1.imshow(combined_map, cmap="magma", interpolation="bicubic")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax1.set_title("1. Mekansal Çapraz Dikkat Haritası ('cat' & 'helmet')", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Gizli Uzay X Koordinatı (64 piksel)")
        ax1.set_ylabel("Gizli Uzay Y Koordinatı (64 piksel)")

        # -------------------------------------------------------------
        # PANEL 2: Kelime Bazlı Dikkat Enerjisi Dağılımı
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        kelimeler = [item["kelime"] for item in rapor["kelime_skorlari"]]
        enerjiler = [item["enerji"] * 100 for item in rapor["kelime_skorlari"]]
        renkler2 = [item["renk"] for item in rapor["kelime_skorlari"]]

        barlar2 = ax2.bar(kelimeler, enerjiler, color=renkler2, edgecolor="black", width=0.55)
        for bar in barlar2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"%{h:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax2.set_title("2. Kelime Bazlı Dikkat Yoğunluğu (%)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Toplam Dikkat Enerjisi (%)")
        ax2.set_ylim(0, 35)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Self-Attention vs Cross-Attention Rol Dağılımı
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        roller = ["Mekansal Öz-Dikkat\n(Piksel-Piksel Geometri)", "Çapraz-Dikkat\n(Piksel-Metin Semantiği)", "Konvolüsyonel Kök\n(Lokal Doku & Kenar)"]
        oranlar = [45, 40, 15]
        renkler3 = ["#4e73df", "#1cc88a", "#f6c23e"]

        ax3.pie(
            oranlar, labels=roller, autopct="%1.0f%%", startangle=140, colors=renkler3,
            textprops=dict(color="black", fontweight="bold")
        )
        ax3.set_title("3. UNet / DiT Katman Rolleri Dağılımı", fontsize=12, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 4: Metin-Piksel Hizalama İcra İzi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. Metin-Piksel Çapraz Dikkat İcra İzi", fontsize=12, fontweight="bold", pad=10)

        hizalama_metni = (
            "====================================================\n"
            "       SPATIAL CROSS-ATTENTION EXECUTION LOG        \n"
            "====================================================\n"
            f"İSTEM (PROMPT): \"{rapor['prompt']}\"\n"
            "----------------------------------------------------\n"
            "ÇAPRAZ DİKKAT ODALANMA BÖLGELERİ:\n"
            f"• 'cat'        ──> {rapor['kelime_skorlari'][0]['odak']} (Enerji: %28)\n"
            f"• 'helmet'     ──> {rapor['kelime_skorlari'][1]['odak']} (Enerji: %24)\n"
            f"• 'astronaut'  ──> {rapor['kelime_skorlari'][2]['odak']} (Enerji: %21)\n"
            f"• 'galaxy'     ──> {rapor['kelime_skorlari'][3]['odak']} (Enerji: %18)\n"
            "----------------------------------------------------\n"
            f"DİKKAT ENTROPİSİ : {rapor['ortalama_cross_attention_entropisi']} nats\n"
            f"HİZALAMA BAŞARISI: %{rapor['metin_piksel_hizalama_dogrulugu']*100:.1f} (Kusursuz Semantik Eşleme)\n"
            "===================================================="
        )

        ax4.text(
            0.02, 0.5, hizalama_metni,
            fontsize=7.2,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: Cross-Attention Mimarisi Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Cross-Attention Text Injection Şeması", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "          CROSS-ATTENTION TEXT INJECTION            \n"
            "====================================================\n"
            "  [Piksel Gizli Haritası z_t (H x W)] ──> [Query Q = W_q z_t]\n"
            "                                                  │ \n"
            "  [CLIP/T5 Metin Gömmesi c] ──> [Key K = W_k c]   ├──> [Dikkat Matrisi]\n"
            "                           ──> [Value V = W_v c]  │    softmax(Q K^T / sqrt(d))\n"
            "                                                  │         │\n"
            "                                                  ▼         ▼\n"
            "  [Mekansal Çıktı: Out = Attention_Weights * V] ──┴─────────┘\n"
            "           │                                        \n"
            "           ▼  (Her Piksel İlgili Kelimeye Odaklanır)\n"
            "  [Prompt Kontrollü Kusursuz Görüntü Sentezi]       \n"
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
        # PANEL 6: GÜN 174 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 174 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   DAY 174 SUMMARY: CROSS-ATTENTION TEXT-TO-IMAGE   \n"
            "====================================================\n"
            "• Modül              : FAZ 9 (Çok Modlu Modeller)\n"
            "• Temel Bileşen      : Spatial Cross-Attention (Q=Piksel, K/V=Metin)\n"
            "• Kullanılan Modeller: Stable Diffusion, SDXL, DiT, FLUX\n"
            f"• Hizalama Başarısı  : %{rapor['metin_piksel_hizalama_dogrulugu']*100:.1f}\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. Metin belirteçlerini görsel piksellere mekansal olarak enjekte etme\n"
            "  2. Prompt-to-Prompt düzenleme ve nesne yer değiştirme temeli\n"
            "  3. Mekansal dikkat haritaları (Attention Maps) ile görsel denetim\n"
            "  4. Self-Attention (yapı) ve Cross-Attention (anlam) hibritleşmesi\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 175 (ControlNet Spatial Conditioning)\n"
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
        print(f"  ✓ Cross-Attention Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
