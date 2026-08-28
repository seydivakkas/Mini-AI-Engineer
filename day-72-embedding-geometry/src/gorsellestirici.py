"""
Temsil Geometrisi ve Boyut İndirgeme Görselleştiricisi
------------------------------------------------------
PCA, t-SNE, UMAP 2D izdüşümleri, SVD tekil değer spektrumu, kosinüs benzerlik
dağılımları ve SWOT matrisini içeren 6 panelli yüksek çözünürlüklü teşhis panosu.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, Optional
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class TemsilGeometrisiGorsellestirici:
    """
    Temsil uzayı geometrisini ve boyut indirgeme analizlerini görselleştiren sınıf.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")
            
        # Renk paleti
        self.sinif_renkleri = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    def olustur_teshis_paneli(
        self,
        X_pca: np.ndarray,
        pca_varyans: np.ndarray,
        X_tsne: np.ndarray,
        tsne_kl: float,
        X_umap: np.ndarray,
        y: np.ndarray,
        izotropi_normal: Dict[str, Any],
        izotropi_cokmus: Dict[str, Any],
        kosinus_metrikleri: Dict[str, Any],
        kayit_yolu: str
    ) -> str:
        """
        6 panelli kapsamlı temsil geometrisi ve boyut indirgeme teşhis panosunu oluşturur.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 72: t-SNE, UMAP Boyut İndirgeme, Temsil Uzayı Geometrisi & İzotropi Analiz Paneli",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        benzersiz_siniflar = np.unique(y)
        palette = sns.color_palette("bright", len(benzersiz_siniflar))

        # -------------------------------------------------------------
        # PANEL 1: PCA 2D İzdüşümü (Lineer Varyans)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        for idx, c in enumerate(benzersiz_siniflar):
            mask = (y == c)
            ax1.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                color=palette[idx], label=f"Sınıf {c}",
                alpha=0.75, edgecolors="none", s=35
            )
        toplam_var = np.sum(pca_varyans) * 100
        ax1.set_title(f"1. PCA 2D İzdüşümü (Açıklanan Varyans: %{toplam_var:.1f})", fontsize=12, fontweight="bold", color="#1a365d")
        ax1.set_xlabel(f"PC1 (%{pca_varyans[0]*100:.1f})", fontsize=10)
        ax1.set_ylabel(f"PC2 (%{pca_varyans[1]*100:.1f})", fontsize=10)
        ax1.legend(loc="upper right", fontsize=8, framealpha=0.8)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 2: t-SNE 2D İzdüşümü (Yerel Manifold & Perplexity)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        for idx, c in enumerate(benzersiz_siniflar):
            mask = (y == c)
            ax2.scatter(
                X_tsne[mask, 0], X_tsne[mask, 1],
                color=palette[idx], label=f"Sınıf {c}",
                alpha=0.75, edgecolors="none", s=35
            )
        ax2.set_title(f"2. t-SNE 2D İzdüşümü (KL Diverjansı: {tsne_kl:.3f})", fontsize=12, fontweight="bold", color="#2b6cb0")
        ax2.set_xlabel("t-SNE Boyut 1", fontsize=10)
        ax2.set_ylabel("t-SNE Boyut 2", fontsize=10)
        ax2.legend(loc="upper right", fontsize=8, framealpha=0.8)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 3: UMAP 2D İzdüşümü (Yerel + Küresel Topoloji)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        for idx, c in enumerate(benzersiz_siniflar):
            mask = (y == c)
            ax3.scatter(
                X_umap[mask, 0], X_umap[mask, 1],
                color=palette[idx], label=f"Sınıf {c}",
                alpha=0.75, edgecolors="none", s=35
            )
        ax3.set_title("3. UMAP 2D İzdüşümü (Riemann Geometrisi & Kosinüs)", fontsize=12, fontweight="bold", color="#2c7a7b")
        ax3.set_xlabel("UMAP Boyut 1", fontsize=10)
        ax3.set_ylabel("UMAP Boyut 2", fontsize=10)
        ax3.legend(loc="upper right", fontsize=8, framealpha=0.8)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 4: SVD Spektrumu ve Boyutsal Çöküş Analizi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        svd_norm = np.array(izotropi_normal["tekil_degerler"])
        svd_cokmus = np.array(izotropi_cokmus["tekil_degerler"])
        eksenler = np.arange(1, len(svd_norm) + 1)
        
        ax4.plot(eksenler, svd_norm / (svd_norm[0] + 1e-12), "o-", color="#2b6cb0", linewidth=2, label=f"Sağlıklı Temsil (İzotropi: {izotropi_normal['izotropi_skoru']:.3f})")
        ax4.plot(eksenler, svd_cokmus / (svd_cokmus[0] + 1e-12), "s--", color="#e53e3e", linewidth=2, label=f"Çökmüş Temsil (İzotropi: {izotropi_cokmus['izotropi_skoru']:.3f})")

        ax4.set_title("4. Tekil Değer Spektrumu & SVD Decay", fontsize=12, fontweight="bold", color="#744210")
        ax4.set_xlabel("Tekil Değer İndeksi (Eksen)", fontsize=10)
        ax4.set_ylabel("Normalize Tekil Değer (σ_i / σ_1)", fontsize=10)
        ax4.set_yscale("log")
        ax4.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax4.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 5: Kosinüs Benzerliği Dağılımı (Sınıf İçi vs Sınıflar Arası)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        sinif_ici = np.array(kosinus_metrikleri["sinif_ici_ornekler"])
        siniflar_arasi = np.array(kosinus_metrikleri["siniflar_arasi_ornekler"])
        
        sns.kdeplot(sinif_ici, ax=ax5, color="#2f855a", fill=True, alpha=0.35, label=f"Sınıf İçi (Ort: {kosinus_metrikleri['sinif_ici_ortalama_kosinus']:.3f})")
        sns.kdeplot(siniflar_arasi, ax=ax5, color="#c53030", fill=True, alpha=0.35, label=f"Sınıflar Arası (Ort: {kosinus_metrikleri['siniflar_arasi_ortalama_kosinus']:.3f})")
        
        ax5.axvline(kosinus_metrikleri["sinif_ici_ortalama_kosinus"], color="#2f855a", linestyle="--", linewidth=1.5)
        ax5.axvline(kosinus_metrikleri["siniflar_arasi_ortalama_kosinus"], color="#c53030", linestyle="--", linewidth=1.5)
        
        ax5.set_title(f"5. Kosinüs Benzerliği Dağılımı (Ayrışma Marjini: {kosinus_metrikleri['ayrisma_marjini']:.3f})", fontsize=12, fontweight="bold", color="#276749")
        ax5.set_xlabel("Kosinüs Benzerliği (Cosine Similarity)", fontsize=10)
        ax5.set_ylabel("Yoğunluk (Density)", fontsize=10)
        ax5.legend(loc="upper left", fontsize=8, framealpha=0.8)
        ax5.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 6: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        
        swot_metni = (
            "           TEMSİL GEOMETRİSİ & BOYUT İNDİRGEME SWOT MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • UMAP hem yerel manifoldları hem de küresel mesafeleri korur.\n"
            "  • SVD İzotropi Analizi, boyutsal çöküşü (collapse) anında yakalar.\n"
            "  • Kosinüs ayrışma marjini, metrik öğrenim kalitesini sayısallaştırır.\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • t-SNE küresel mesafeleri korumaz (kümeler arası uzaklık anlamsızdır).\n"
            "  • t-SNE O(N^2) hesaplama karmaşıklığı ile büyük veride yavaştır.\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • SimCLR ve SupCon öncesi temsil kalitesini denetleme standardı.\n"
            "  • LLM ve Vision Transformer latent uzaylarında anizotropi teşhisi.\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • Aşırı perplexity/neighbor seçimi ile yapay kümelenme yanılsaması.\n"
            "  • Anizotropik koni oluşumu sonucu kosinüs benzerliğinin bozulması."
        )
        
        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#fffaf0", edgecolor="#dd6b20", linewidth=1.8)
        )
        ax6.set_title("6. Boyut İndirgeme ve Geometri SWOT Matrisi", fontsize=12, fontweight="bold", color="#9c4221")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
