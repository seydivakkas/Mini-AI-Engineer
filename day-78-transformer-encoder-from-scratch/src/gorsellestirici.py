"""
Transformer Encoder Teşhis ve Analiz Görselleştiricisi
------------------------------------------------------
6 panelli yüksek çözünürlüklü Transformer Encoder mimari ve gradyan teşhis panosu.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


class EncoderGorsellestirici:
    """
    Transformer Encoder bileşenlerini ve gradyan kararlılığını görselleştiren sınıf.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")

    def olustur_teshis_paneli(
        self,
        pe_matrisi: np.ndarray,
        pre_ln_gradyanlar: List[float],
        post_ln_gradyanlar: List[float],
        gelu_ciktilari: np.ndarray,
        relu_ciktilari: np.ndarray,
        katman_benzerlikleri: List[float],
        kayit_yolu: str
    ) -> str:
        """
        6 panelli kapsamlı Transformer Encoder teşhis panosunu oluşturur.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 78: Sıfırdan Transformer Encoder Bloğu (Pre-LN, Pozisyonel Kodlama, FFN & Residual Paneli)",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        # -------------------------------------------------------------
        # PANEL 1: Pre-LN vs Post-LN Mimari Karşılaştırma Şeması
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        
        sema_metni = (
            "      TRANSFORMER ENCODER BLOK MİMARİSİ\n"
            "─────────────────────────────────────────────────────────\n"
            "  1. PRE-LAYER NORM (Modern Standart - ViT / GPT):\n"
            "     x = x + MHSA( LayerNorm(x) )\n"
            "     x = x + FFN( LayerNorm(x) )\n"
            "     • Saf residual omurga (Identity gradient flow).\n"
            "     • 100+ katmanda Warmup olmadan kararlı eğitim.\n\n"
            "  2. POST-LAYER NORM (Orijinal Vaswani 2017):\n"
            "     x = LayerNorm( x + MHSA(x) )\n"
            "     x = LayerNorm( x + FFN(x) )\n"
            "     • Derin katmanlarda gradyan patlaması riski.\n\n"
            "  3. RESIDUAL FFN BLOĞU:\n"
            "     • x -> W1 (D -> 4D) -> GELU -> Dropout -> W2 (4D -> D)"
        )
        ax1.text(
            0.5, 0.5, sema_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#ebf8ff", edgecolor="#3182ce", linewidth=1.8)
        )
        ax1.set_title("1. Pre-LN vs Post-LN Mimari Tasarımı", fontsize=12, fontweight="bold", color="#2b6cb0")

        # -------------------------------------------------------------
        # PANEL 2: Sinüzoidal Pozisyonel Kodlama Isı Haritası
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        sns.heatmap(pe_matrisi, ax=ax2, cmap="magma", cbar=True)
        ax2.set_title(f"2. Sinüzoidal Pozisyonel Kodlama Matrisi ({pe_matrisi.shape[0]} Poz x {pe_matrisi.shape[1]} Dim)", fontsize=12, fontweight="bold", color="#2c5282")
        ax2.set_xlabel("Gömülme Boyutu İndeksi (0 ila 63)", fontsize=10)
        ax2.set_ylabel("Token Pozisyon İndeksi (0 ila 31)", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Pre-LN vs Post-LN Gradyan Norm Yayılımı
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        katman_ind = np.arange(len(pre_ln_gradyanlar))
        w = 0.35
        ax3.bar(katman_ind - w/2, pre_ln_gradyanlar, w, label="Pre-LN (Kararlı / Düzgün)", color="#38a169", alpha=0.9)
        ax3.bar(katman_ind + w/2, post_ln_gradyanlar, w, label="Post-LN (Dengesiz / Uç Değerler)", color="#e53e3e", alpha=0.85)

        ax3.set_xticks(katman_ind)
        ax3.set_xticklabels([f"Katman {i+1}" for i in range(len(pre_ln_gradyanlar))], fontsize=9)
        ax3.set_ylabel("Girdi Gradyan Normu (Grad Norm)", fontsize=10)
        ax3.set_title("3. Pre-LN vs Post-LN Gradyan Kararlılığı", fontsize=12, fontweight="bold", color="#276749")
        ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 4: FFN Aktivasyon Dağılımı (GELU vs ReLU)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        sns.kdeplot(gelu_ciktilari.flatten(), ax=ax4, color="#3182ce", linewidth=2.5, label="GELU (Pürüzsüz / ViT Standartı)")
        sns.kdeplot(relu_ciktilari.flatten(), ax=ax4, color="#dd6b20", linewidth=2.5, linestyle="--", label="ReLU (Sert Kırılma)")

        ax4.set_title("4. FFN Gizli Katman Aktivasyon Dağılımı", fontsize=12, fontweight="bold", color="#2c7a7b")
        ax4.set_xlabel("Aktivasyon Değeri", fontsize=10)
        ax4.set_ylabel("Yoğunluk (Density)", fontsize=10)
        ax4.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
        ax4.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 5: Katmanlar Boyunca Temsil Benzerliği
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        x_steps = [f"L{i} -> L{i+1}" for i in range(len(katman_benzerlikleri))]
        ax5.plot(x_steps, katman_benzerlikleri, "o-", color="#805ad5", linewidth=2.5, markersize=8)
        for i, val in enumerate(katman_benzerlikleri):
            ax5.text(i, val + 0.01, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

        ax5.set_ylim(0.0, 1.1)
        ax5.set_ylabel("Kosinüs Benzerliği", fontsize=10)
        ax5.set_title("5. Katmanlar Arası Temsil Akışı & Değişimi", fontsize=12, fontweight="bold", color="#4a5568")
        ax5.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 6: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        
        swot_metni = (
            "         TRANSFORMER ENCODER BLOĞU SWOT MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • Pre-LN ile yüzlerce katmanda bile pürüzsüz gradyan akışı.\n"
            "  • Residual bağlantılar ile özellik bozulmasını (Degradation) önleme.\n"
            "  • GELU aktivasyonlu 4x FFN ile zengin doğrusal olmayan kapasite.\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • FFN katmanı model parametrelerinin ~%66'sını oluşturur (Bellek yükü).\n"
            "  • Pozisyonel kodlama dizinin maksimum uzunluğuna bağlı kalabilir.\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • Vision Transformer (ViT) omurgasının doğrudan temel yapı taşı.\n"
            "  • SwiGLU / RMSNorm entegrasyonlarıyla çıkarım hızını %25 artırma.\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • Yanlış LayerNorm konfigürasyonunda (Post-LN) erken eğitim ıraksaması."
        )
        
        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f7fafc", edgecolor="#4a5568", linewidth=1.8)
        )
        ax6.set_title("6. Transformer Encoder SWOT Karar Matrisi", fontsize=12, fontweight="bold", color="#2d3748")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
