"""
SimCLR Kontrastif Temsil Öğrenimi Görselleştiricisi
---------------------------------------------------
Çift görünümlü artırma örnekleri, NT-Xent kayıp trajektorisi, temsil uzayı izdüşümü,
sıcaklık katsayısı analizi, kosinüs marjini gelişimi ve SWOT matrisini içeren 6 panelli pano.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any, Optional
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.decomposition import PCA


class SimCLRGorsellestirici:
    """
    SimCLR eğitim dinamiklerini ve temsil geometrisini görselleştiren sınıf.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")

    def olustur_teshis_paneli(
        self,
        ornek_ciftler: List[Tuple[np.ndarray, np.ndarray]],
        egitim_gecmisi: Dict[str, List[float]],
        temsiller_2d: np.ndarray,
        etiketler: np.ndarray,
        kayit_yolu: str
    ) -> str:
        """
        6 panelli SimCLR kontrastif temsil analiz panosunu oluşturur ve kaydeder.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 73: Sıfırdan SimCLR Temsil Öğrenimi, Artırma Çiftleri & NT-Xent (InfoNCE) Paneli",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        epochs = egitim_gecmisi["epoch"]

        # -------------------------------------------------------------
        # PANEL 1: Çift Görünümlü Artırma Örnekleri (Positive Views)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        # 3 örnek çift göster (v1 üstte, v2 altta veya yan yana grid)
        ax1.axis("off")
        grid_img = np.zeros((64, 32 * len(ornek_ciftler), 3))
        for idx, (v1, v2) in enumerate(ornek_ciftler):
            # v1 (üst), v2 (alt)
            grid_img[0:32, idx*32:(idx+1)*32, :] = np.clip(v1, 0, 1)
            grid_img[32:64, idx*32:(idx+1)*32, :] = np.clip(v2, 0, 1)
            
        ax1.imshow(grid_img)
        ax1.set_title("1. Stokastik Artırma Çiftleri (Üst: Görünüm 1, Alt: Görünüm 2)", fontsize=12, fontweight="bold", color="#1a365d")

        # -------------------------------------------------------------
        # PANEL 2: NT-Xent Kayıp Eğrisi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(epochs, egitim_gecmisi["loss"], "o-", color="#d62728", linewidth=2.5, label="NT-Xent (InfoNCE) Kaybı")
        ax2.plot(epochs, egitim_gecmisi["alignment_loss"], "s--", color="#2ca02c", linewidth=1.8, label="Alignment Hatası (||z1-z2||²)")
        ax2.set_title(f"2. NT-Xent Kayıp Trajektorisi (Son: {egitim_gecmisi['loss'][-1]:.4f})", fontsize=12, fontweight="bold", color="#9b2c2c")
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("Kayıp (Loss)", fontsize=10)
        ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 3: Temsil Uzayı (h) 2D İzdüşümü (PCA)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        benzersiz_siniflar = np.unique(etiketler)
        palette = sns.color_palette("bright", len(benzersiz_siniflar))
        for idx, c in enumerate(benzersiz_siniflar):
            mask = (etiketler == c)
            ax3.scatter(
                temsiller_2d[mask, 0], temsiller_2d[mask, 1],
                color=palette[idx], label=f"Sınıf {int(c)}",
                alpha=0.75, edgecolors="none", s=40
            )
        ax3.set_title("3. Etiketsiz Öğrenilen Temsil Uzayı (h Projeksiyonu)", fontsize=12, fontweight="bold", color="#2c7a7b")
        ax3.set_xlabel("Temsil Boyut 1", fontsize=10)
        ax3.set_ylabel("Temsil Boyut 2", fontsize=10)
        ax3.legend(loc="upper right", fontsize=8, framealpha=0.8)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 4: Sıcaklık Katsayısı (τ) Sertlik Analizi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        neg_sim = np.linspace(-1.0, 1.0, 200)
        for tau, renk in [(0.1, "#e53e3e"), (0.2, "#dd6b20"), (0.5, "#3182ce"), (1.0, "#718096")]:
            # Ceza ağırlığı: exp(sim / tau)
            ceza = np.exp(neg_sim / tau)
            ceza_norm = ceza / np.max(ceza)
            ax4.plot(neg_sim, ceza_norm, linewidth=2, color=renk, label=f"τ = {tau} (Sertlik)")
            
        ax4.set_title("4. Sıcaklık Parametresi (τ) ve Zor Negatif Cezalandırma", fontsize=12, fontweight="bold", color="#744210")
        ax4.set_xlabel("Negatif Çift Kosinüs Benzerliği (sim(z_i, z_k))", fontsize=10)
        ax4.set_ylabel("Normalize Ceza Ağırlığı", fontsize=10)
        ax4.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax4.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 5: Kosinüs Benzerliği ve Marjin Gelişimi
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.plot(epochs, egitim_gecmisi["pozitif_kosinus"], "o-", color="#2f855a", linewidth=2, label="Pozitif Çiftler (sim(z1, z2))")
        ax5.plot(epochs, egitim_gecmisi["negatif_kosinus"], "x--", color="#c53030", linewidth=2, label="Negatif Çiftler (sim(z_i, z_k))")
        ax5.plot(epochs, egitim_gecmisi["kosinus_marjini"], "^-.", color="#3182ce", linewidth=2.5, label="Ayrışma Marjini (Poz - Neg)")
        
        son_marjin = egitim_gecmisi["kosinus_marjini"][-1]
        ax5.set_title(f"5. Kosinüs Benzerliği ve Marjin Gelişimi (Son Marjin: {son_marjin:.3f})", fontsize=12, fontweight="bold", color="#276749")
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
            "              SimCLR & KONTRASTİF ÖĞRENİM SWOT MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • Tek bir etiket olmadan ImageNet seviyesinde temsil gücü.\n"
            "  • Non-lineer Projeksiyon Kafası ile downstream doğruluğunda +%10 sıçrama.\n"
            "  • NT-Xent kaybı ile pozitifleri çekerken negatifleri homojen iter.\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • Büyük batch boyutu gereksinimi (N=4096) - Çok GPU belleği ister.\n"
            "  • Yanlış artırma politikası seçildiğinde temsil kalitesi çöker.\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • Tıp ve sanayi gibi etiket maliyeti astronomik alanlarda devrim.\n"
            "  • Linear Probing ile sadece %1 etiketle tam denetimli modeli yakalama.\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • Negatif çiftlerin içinde aslında aynı sınıftan olan örnekler (False Negatives).\n"
            "  • Sıcaklık katsayısı τ çok küçük seçilirse gradyan patlaması."
        )
        
        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#fffaf0", edgecolor="#dd6b20", linewidth=1.8)
        )
        ax6.set_title("6. SimCLR Mimarisi SWOT Matrisi", fontsize=12, fontweight="bold", color="#9c4221")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
