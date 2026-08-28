"""
SupCon (Supervised Contrastive) Öğrenim Görselleştiricisi
---------------------------------------------------------
6 panelli yüksek çözünürlüklü görsel teşhis panosu üreten modül.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any, Tuple
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class SupConGorsellestirici:
    """
    SupCon Stage 1 ve Stage 2 eğitim dinamiklerini görselleştiren sınıf.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")

    def olustur_teshis_paneli(
        self,
        ornek_ciftler: List[Tuple[np.ndarray, np.ndarray, int]],
        stage1_gecmisi: Dict[str, List[float]],
        stage2_gecmisi: Dict[str, List[float]],
        temsiller_2d: np.ndarray,
        etiketler: np.ndarray,
        kayit_yolu: str
    ) -> str:
        """
        6 panelli SupCon teşhis panosunu oluşturur ve diske kaydeder.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 74: Etiketli Veride Supervised Contrastive (SupCon) Kaybı ile Sınıf Ayrıştırma Paneli",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        epochs1 = stage1_gecmisi["epoch"]
        epochs2 = stage2_gecmisi["epoch"]

        # -------------------------------------------------------------
        # PANEL 1: Çift Görünümlü Artırma Örnekleri (Görünüm 1 & 2)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        grid_img = np.zeros((64, 32 * len(ornek_ciftler), 3))
        for idx, (v1, v2, c) in enumerate(ornek_ciftler):
            grid_img[0:32, idx*32:(idx+1)*32, :] = np.clip(v1, 0, 1)
            grid_img[32:64, idx*32:(idx+1)*32, :] = np.clip(v2, 0, 1)
            
        ax1.imshow(grid_img)
        ax1.set_title("1. SupCon Artırma Çiftleri (Üst: Görünüm 1, Alt: Görünüm 2)", fontsize=12, fontweight="bold", color="#1a365d")

        # -------------------------------------------------------------
        # PANEL 2: Stage 1 SupCon Kayıp Eğrisi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(epochs1, stage1_gecmisi["loss"], "o-", color="#d62728", linewidth=2.5, label="SupCon Kaybı (τ=0.1)")
        ax2.set_title(f"2. Stage 1 SupCon Kayıp Trajektorisi (Son: {stage1_gecmisi['loss'][-1]:.4f})", fontsize=12, fontweight="bold", color="#9b2c2c")
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("SupCon Kayıp", fontsize=10)
        ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 3: SupCon ile Öğrenilen Temsil Uzayı (h Projeksiyonu - PCA)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        benzersiz_siniflar = np.unique(etiketler)
        palette = sns.color_palette("bright", len(benzersiz_siniflar))
        for idx, c in enumerate(benzersiz_siniflar):
            mask = (etiketler == c)
            ax3.scatter(
                temsiller_2d[mask, 0], temsiller_2d[mask, 1],
                color=palette[idx], label=f"Sınıf {int(c)}",
                alpha=0.8, edgecolors="none", s=45
            )
        ax3.set_title("3. SupCon ile Öğrenilen Temsil Uzayı (Mükemmel Sınıf Kümeleri)", fontsize=12, fontweight="bold", color="#2c7a7b")
        ax3.set_xlabel("Temsil Boyut 1", fontsize=10)
        ax3.set_ylabel("Temsil Boyut 2", fontsize=10)
        ax3.legend(loc="upper right", fontsize=8, framealpha=0.8)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 4: Stage 2 Doğrusal Sınıflandırma Doğruluğu (Linear Probing)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.plot(epochs2, stage2_gecmisi["dogruluk"], "s-", color="#2b6cb0", linewidth=2.5, label="Doğrulama Doğruluğu (Val Acc %)")
        son_acc = stage2_gecmisi["dogruluk"][-1]
        ax4.set_title(f"4. Stage 2 Linear Probing Doğruluğu (Son: %{son_acc:.2f})", fontsize=12, fontweight="bold", color="#2b6cb0")
        ax4.set_xlabel("Epoch", fontsize=10)
        ax4.set_ylabel("Doğruluk (%)", fontsize=10)
        ax4.legend(loc="lower right", fontsize=9, framealpha=0.9)
        ax4.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 5: Kosinüs Benzerliği ve Marjin Gelişimi
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.plot(epochs1, stage1_gecmisi["sinif_ici_kosinus"], "o-", color="#2f855a", linewidth=2, label="Sınıf İçi Pozitifler (Aynı Sınıf)")
        ax5.plot(epochs1, stage1_gecmisi["siniflar_arasi_kosinus"], "x--", color="#c53030", linewidth=2, label="Sınıflar Arası Negatifler (Farklı Sınıf)")
        ax5.plot(epochs1, stage1_gecmisi["ayrisma_marjini"], "^-.", color="#3182ce", linewidth=2.5, label="Ayrışma Marjini (İçi - Arası)")
        
        son_marjin = stage1_gecmisi["ayrisma_marjini"][-1]
        ax5.set_title(f"5. Sınıf İçi vs Sınıflar Arası Ayrışma (Marjin: {son_marjin:.3f})", fontsize=12, fontweight="bold", color="#276749")
        ax5.set_xlabel("Epoch", fontsize=10)
        ax5.set_ylabel("Kosinüs Benzerliği", fontsize=10)
        ax5.legend(loc="center left", fontsize=8.5, framealpha=0.9)
        ax5.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 6: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        
        swot_metni = (
            "           SUPERVISED CONTRASTIVE (SupCon) SWOT MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • Cross-Entropy'ye göre etiket gürültüsüne (label noise) karşı aşırı dayanıklı.\n"
            "  • SimCLR'daki 'False Negative' sorununu sınıf etiketleri ile tamamen çözer.\n"
            "  • Temsil uzayında sınıflar arasında devasa bir geometrik marjin açar.\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • İki aşamalı eğitim gerektirir (Stage 1: SupCon + Stage 2: Linear Probing).\n"
            "  • Batch içinde her sınıftan birden fazla örnek bulunması zorunludur.\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • Az örnekli öğrenme (Few-shot) ve Dağılım Dışı (OOD) tespiti.\n"
            "  • Tıbbi görüntüleme ve yüz tanıma sistemlerinde sınıf ayrımı.\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • Çok küçük batch boyutu kullanıldığında pozitif eşleşme bulunamaması."
        )
        
        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#fffaf0", edgecolor="#dd6b20", linewidth=1.8)
        )
        ax6.set_title("6. SupCon Mimarisi SWOT Matrisi", fontsize=12, fontweight="bold", color="#9c4221")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
