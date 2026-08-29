"""
LoRA ve DreamBooth Teşhis Panosu Görselleştirici Modülü (Day 176 - FAZ 9).
6 panelli LoRA Rank Kıyası (r=4..64 vs Boyut/Kalite), Parametre Tasarrufu, DreamBooth Sınıf Koruma Kaybı İzi, Çoklu LoRA Karışım İzi, Mimarisi ve Özet Kartı.
"""

import os
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np


class LoRAGorsellestirici:
    """LoRA & DreamBooth İnce Ayar Teşhis Panosu Üreticisi."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        rapor: Dict[str, Any],
        kayit_yolu: str = "ciktilar/lora_diffusion_finetuning_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(23, 13.5))
        fig.suptitle(
            "GÜN 176 (FAZ 9): Difüzyon Modellerinde LoRA (Low-Rank Adaptation) & DreamBooth ile Özel Nesne ve Sanat Stili Öğretimi",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: LoRA Rank (r) vs Model Dosya Boyutu ve Sadakat Skoru
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ranks = [f"r={item['r']}" for item in rapor["rank_deneyleri"]]
        boyutlar = [item["dosya_mb"] for item in rapor["rank_deneyleri"]]
        sadakatler = [item["sadakat"] * 100 for item in rapor["rank_deneyleri"]]

        ax1_twin = ax1.twinx()
        bar1 = ax1.bar(ranks, boyutlar, color="#4e73df", alpha=0.7, width=0.4, label="LoRA Dosya Boyutu (MB)")
        line1 = ax1_twin.plot(ranks, sadakatler, color="#e74a3b", marker="o", linewidth=2.5, label="Özne Sadakati (%)")

        ax1.set_title("1. LoRA Rank Seçimi: Dosya Boyutu vs Sadakat", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Dosya Boyutu (MB)", color="#4e73df")
        ax1_twin.set_ylabel("Özne Sadakati (%)", color="#e74a3b")
        ax1_twin.set_ylim(80, 102)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Model Parametre ve Bellek Tasarrufu (Full Finetune vs LoRA)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        modeller = ["Full Finetune\n(Tüm UNet Checkpoint)", "LoRA İnce Ayar\n(r=16 Düşük Dereceli)"]
        param_mb = [4200.0, 36.4]
        renkler2 = ["#e74a3b", "#1cc88a"]

        barlar2 = ax2.bar(modeller, param_mb, color=renkler2, edgecolor="black", width=0.45)
        for bar in barlar2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 80, f"{h:.1f} MB", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax2.set_title("2. Depolama & Parametre Tasarrufu (~115 Kat)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Dosya Boyutu (MB)")
        ax2.set_ylim(0, 4800)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: DreamBooth Sınıf Koruma Kaybı (Prior Preservation)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        adimlar = np.linspace(0, 1000, 50)
        loss_inst = 0.5 * np.exp(-adimlar / 250) + 0.02
        loss_prior = 0.4 * np.exp(-adimlar / 350) + 0.015

        ax3.plot(adimlar, loss_inst, label="Instance Loss ('sks robot')", color="#4e73df", linewidth=2.2)
        ax3.plot(adimlar, loss_prior, label="Prior Loss (Genel 'robot' Sınıfı)", color="#1cc88a", linewidth=2.2, linestyle="--")

        ax3.set_title("3. DreamBooth Çift Kayıp Yakınsaması", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Eğitim Adımı")
        ax3.set_ylabel("MSE Gürültü Kestirim Kaybı")
        ax3.legend(loc="upper right", frameon=True)
        ax3.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Çoklu LoRA Karışımı ve İcra İzi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. Çoklu LoRA Ağırlık Birleştirme İzi", fontsize=12, fontweight="bold", pad=10)

        lora_metni = (
            "====================================================\n"
            "       MULTI-LORA BLENDING & DREAMBOOTH LOG         \n"
            "====================================================\n"
            f"ÖZEL KAVRAM : {rapor['hedef_kavram']}\n"
            "----------------------------------------------------\n"
            "YÜKLENEN LORA AĞIRLIKLARI:\n"
            "• LoRA 1: 'Cyberpunk Sanat Stili'   (Ağırlık: 0.7)\n"
            "• LoRA 2: 'sks robot Karakteri'     (Ağırlık: 1.0)\n"
            "----------------------------------------------------\n"
            "AĞIRLIK BİRLEŞTİRME FORMÜLÜ:\n"
            "  W_merged = W_0 + 0.7*(B_1*A_1) + 1.0*(B_2*A_2)\n"
            "----------------------------------------------------\n"
            f"ÖZNE SADAKATİ  : %{rapor['sinif_koruma_skoru']*100:.1f}\n"
            f"DİL KAYMASI    : %0.0 (Genel Sınıf Bilgisi Korundu)\n"
            f"KAZANÇ         : {rapor['dosya_boyut_kazanci']}\n"
            "===================================================="
        )

        ax4.text(
            0.02, 0.5, lora_metni,
            fontsize=7.2,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: LoRA Cross-Attention Enjeksiyon Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. LoRA Düşük Dereceli Matris Mimarisi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "         LORA LOW-RANK MATRIX DECOMPOSITION         \n"
            "====================================================\n"
            "  Giriş Vektörü (x) ─────────────────────────┐      \n"
            "           │                                 │      \n"
            "           ▼                                 ▼      \n"
            "  [Dondurulmuş W_0 (d x k)]        [LoRA_A (r x k)] \n"
            "           │ (Kilitli Ağırlık)               │      \n"
            "           │                                 ▼      \n"
            "           │                       [LoRA_B (d x r)] \n"
            "           │                                 │      \n"
            "           ▼                                 ▼      \n"
            "  [Orijinal Çıktı: x W_0] ──── (+) ──── [(x A^T B^T) * (a/r)]\n"
            "                                │                   \n"
            "                                ▼                   \n"
            "  [Yeni Özelleştirilmiş Çıktı: y = x (W_0 + Delta_W)]\n"
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
        # PANEL 6: GÜN 176 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 176 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   DAY 176 SUMMARY: LORA & DREAMBOOTH DIFFUSION     \n"
            "====================================================\n"
            "• Modül              : FAZ 9 (Çok Modlu Modeller)\n"
            "• Temel Mimariler    : LoRA (Hu et al.), DreamBooth (Ruiz et al.)\n"
            "• İdeal Rank Değeri  : r = 8 - 16 (Alpha = 16 - 32)\n"
            f"• Boyut Tasarrufu    : {rapor['dosya_boyut_kazanci']}\n"
            f"• Sadakat Oranı      : %{rapor['sinif_koruma_skoru']*100:.1f}\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. 4GB model kopyalamak yerine 30MB LoRA dosyası paylaşma\n"
            "  2. Sadece 3-5 fotoğrafla özel karakter/ürün öğretimi\n"
            "  3. Prior Preservation Loss ile dil kaymasını (forgetting) önleme\n"
            "  4. Birden fazla LoRA'yı gerçek zamanlı ağırlıkla birleştirme\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 177 (Diffusion Transformers - DiT)\n"
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
        print(f"  ✓ LoRA & DreamBooth Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
