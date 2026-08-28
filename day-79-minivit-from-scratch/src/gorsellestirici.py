"""
Mini Vision Transformer Teşhis ve Görselleştirme Panosu
-------------------------------------------------------
6 panelli yüksek çözünürlüklü MiniViT mimari ve dikkat yayılım panosu.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any, Tuple
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F


class MiniViTGorsellestirici:
    """
    MiniViT mimarisini, yama ayrıştırmasını, Attention Rollout'u ve parametre dağılımını görselleştiren sınıf.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")

    def olustur_teshis_paneli(
        self,
        orijinal_gorsel: np.ndarray,
        dikkat_isi_haritasi: np.ndarray,
        pos_embed_tensor: torch.Tensor,
        parametre_dagilimi: Dict[str, int],
        kayit_yolu: str
    ) -> str:
        """
        6 panelli kapsamlı MiniViT teşhis panosunu oluşturur.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 79: Sıfırdan Mini Vision Transformer (Patch Projeksiyonu, [CLS] Token, Encoder & Attention Rollout Paneli)",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        # -------------------------------------------------------------
        # PANEL 1: MiniViT Mimari Akış Şeması
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        
        mimari_metin = (
            "          MINI VISION TRANSFORMER (MiniViT) MİMARİSİ\n"
            "─────────────────────────────────────────────────────────────\n"
            "  1. GİRDİ GÖRSELİ: x ∈ ℝ^(B × 3 × 32 × 32)\n\n"
            "  2. YAMA GÖMÜLME (Patch Embedding - P=4x4):\n"
            "     • N = (32/4) × (32/4) = 64 yama\n"
            "     • Conv2D(3 -> 64, kernel=4, stride=4) ──> x_p ∈ ℝ^(B × 64 × 64)\n\n"
            "  3. [CLS] TOKEN & POZİSYONEL GÖMÜLME:\n"
            "     • z_0 = [CLS; x_p] + E_pos ∈ ℝ^(B × 65 × 64)\n\n"
            "  4. L=4 KATMANLI PRE-LN TRANSFORMER ENCODER:\n"
            "     • z_l = PreLN_EncoderBlock(z_(l-1))\n\n"
            "  5. MLP SINIFLANDIRMA KAFASI:\n"
            "     • y_hat = Linear( LayerNorm(z_L[:, 0]) ) ──> (B × 10 Sınıf)"
        )
        ax1.text(
            0.5, 0.5, mimari_metin,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#ebf8ff", edgecolor="#3182ce", linewidth=1.8)
        )
        ax1.set_title("1. MiniViT Uçtan Uca Hesaplama Akışı", fontsize=12, fontweight="bold", color="#2b6cb0")

        # -------------------------------------------------------------
        # PANEL 2: Görselin 4x4 Yamalara Bölünmesi (Patch Grid)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        img = np.transpose(orijinal_gorsel, (1, 2, 0)) # (H, W, C)
        ax2.imshow(np.clip(img, 0.0, 1.0))
        
        # 4x4 ızgara çizgileri çiz
        H, W, _ = img.shape
        for x in range(0, W, 4):
            ax2.axvline(x - 0.5, color="white", linestyle="--", alpha=0.7, linewidth=1.2)
        for y in range(0, H, 4):
            ax2.axhline(y - 0.5, color="white", linestyle="--", alpha=0.7, linewidth=1.2)

        ax2.set_title(f"2. 2D Görselin 4x4 Yamalara Ayrıştırılması (N=64)", fontsize=12, fontweight="bold", color="#2c5282")
        ax2.axis("off")

        # -------------------------------------------------------------
        # PANEL 3: [CLS] Token Attention Rollout Isı Haritası
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.imshow(np.clip(img, 0.0, 1.0))
        # Isı haritasını yarı saydam üstüne bindir
        im_heat = ax3.imshow(dikkat_isi_haritasi, cmap="jet", alpha=0.55)
        plt.colorbar(im_heat, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_title("3. [CLS] Token Attention Rollout (Görsel Odak)", fontsize=12, fontweight="bold", color="#c53030")
        ax3.axis("off")

        # -------------------------------------------------------------
        # PANEL 4: Öğrenilebilir Pozisyonel Gömülme Benzerlik Matrisi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        pe = pos_embed_tensor.squeeze(0) # (65, D)
        pe_norm = F.normalize(pe, p=2, dim=1)
        sim_mat = torch.matmul(pe_norm, pe_norm.T).detach().cpu().numpy()

        sns.heatmap(sim_mat, ax=ax4, cmap="magma", cbar=True)
        ax4.set_title("4. Pozisyonel Gömülmelerin Kosinüs Benzerliği (65x65)", fontsize=12, fontweight="bold", color="#2c7a7b")
        ax4.set_xlabel("Token Pozisyonu (0: CLS, 1-64: Yamalar)", fontsize=9)
        ax4.set_ylabel("Token Pozisyonu", fontsize=9)

        # -------------------------------------------------------------
        # PANEL 5: MiniViT Model Parametre Dağılımı
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        labels = list(parametre_dagilimi.keys())
        counts = list(parametre_dagilimi.values())
        colors = ["#4299e1", "#ed8936", "#48bb78", "#9f7aea"]

        wedges, texts, autotexts = ax5.pie(
            counts, labels=labels, autopct="%1.1f%%",
            startangle=140, colors=colors, textprops=dict(fontweight="bold")
        )
        toplam_param = sum(counts)
        ax5.set_title(f"5. Parametre Dağılımı (Toplam: {toplam_param:,} Parametre)", fontsize=12, fontweight="bold", color="#4a5568")

        # -------------------------------------------------------------
        # PANEL 6: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        
        swot_metni = (
            "           MINI VISION TRANSFORMER (MiniViT) SWOT MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • İlk katmandan itibaren küresel alıcı alan (Global Receptive Field).\n"
            "  • [CLS] token ile dinamik ve açıklanabilir dikkat haritaları (Rollout).\n"
            "  • Evrişimli olmayan saf dizi tabanlı modern mimari.\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • Küçük veri setlerinde CNN'ler kadar hızlı tümevarımsal genelleme yapamaz.\n"
            "  • Yüksek çözünürlükte yama sayısı arttıkça karesel bellek ihtiyacı.\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • Masked Autoencoder (MAE) ve DINO ile self-supervised ön eğitim.\n"
            "  • LoRA / Adapter ile düşük maliyetli fine-tuning uyumluluğu.\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • Güçlü veri artırma (Mixup/CutMix) ve regülarizasyon olmadan aşırı uydurma."
        )
        
        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f7fafc", edgecolor="#4a5568", linewidth=1.8)
        )
        ax6.set_title("6. MiniViT Mimari SWOT Karar Matrisi", fontsize=12, fontweight="bold", color="#2d3748")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
