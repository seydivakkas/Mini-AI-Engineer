"""
Temsil Benchmark Görselleştiricisi
----------------------------------
6 panelli yüksek çözünürlüklü temsil kalitesi teşhis panosu üreten modül.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any, Tuple
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class BenchmarkGorsellestirici:
    """
    Temsil kalitesi benchmark sonuçlarını görselleştiren sınıf.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")

    def olustur_teshis_paneli(
        self,
        egitilmis_sonuclar: Dict[str, Any],
        rastgele_sonuclar: Dict[str, Any],
        few_shot_egrisi: Dict[str, List[float]],
        gomulmeler_2d: np.ndarray,
        etiketler: np.ndarray,
        kayit_yolu: str
    ) -> str:
        """
        6 panelli kapsamlı temsil kalitesi teşhis panosunu oluşturur.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 76: Temsil Kalitesi Değerlendirmesi (Linear Probing, k-NN & Manifold Geometrisi Paneli)",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        # -------------------------------------------------------------
        # PANEL 1: Değerlendirme Protokolleri Karşılaştırma Şeması
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        
        proto_sema = (
            "       TEMSİL KALİTESİ DEĞERLENDİRME STANDARTLARI\n"
            "─────────────────────────────────────────────────────────\n"
            "  1. LINEAR PROBING (Altın Standart):\n"
            "     • Omurga f(x) dondurulur (Freeze).\n"
            "     • Tek bir lineer katman (W*h + b) eğitilir.\n"
            "     • Amaç: Temsilin doğrusal ayrışabilirliğini ölçmek.\n\n"
            "  2. NON-PARAMETRIC k-NN EVALUATION:\n"
            "     • SIFIR EĞİTİM (0 Epoch, Zero-parameter).\n"
            "     • Sıcaklık ölçekli ağırlıklı oylama (exp(sim/τ)).\n"
            "     • Amaç: Yerel manifold topolojisini doğrudan test etmek.\n\n"
            "  3. FEW-SHOT DATA-EFFICIENCY:\n"
            "     • %1 ve %10 etiket ile Linear Probing.\n"
            "     • Amaç: Az veriyle transfer kabiliyetini ölçmek."
        )
        ax1.text(
            0.5, 0.5, proto_sema,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#ebf8ff", edgecolor="#3182ce", linewidth=1.8)
        )
        ax1.set_title("1. Değerlendirme Protokolleri Metodolojisi", fontsize=12, fontweight="bold", color="#2b6cb0")

        # -------------------------------------------------------------
        # PANEL 2: Linear Probing vs k-NN Doğruluk Karşılaştırması
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        metrikler = ["Linear %100", "k-NN (k=1)", "k-NN (k=5)", "k-NN (k=20)"]
        egitilmis_val = [
            egitilmis_sonuclar["linear_probe_100"],
            egitilmis_sonuclar["knn_k_1"],
            egitilmis_sonuclar["knn_k_5"],
            egitilmis_sonuclar["knn_k_20"]
        ]
        rastgele_val = [
            rastgele_sonuclar["linear_probe_100"],
            rastgele_sonuclar["knn_k_1"],
            rastgele_sonuclar["knn_k_5"],
            rastgele_sonuclar["knn_k_20"]
        ]

        x_ind = np.arange(len(metrikler))
        w = 0.35
        ax2.bar(x_ind - w/2, egitilmis_val, w, label="Eğitilmiş Model (Kontrastif)", color="#2b6cb0", alpha=0.9)
        ax2.bar(x_ind + w/2, rastgele_val, w, label="Rastgele Ağırlıklar (Baseline)", color="#a0aec0", alpha=0.8)

        ax2.set_xticks(x_ind)
        ax2.set_xticklabels(metrikler, fontsize=9)
        ax2.set_ylabel("Top-1 Doğruluk (%)", fontsize=10)
        ax2.set_title(f"2. Linear Probe vs k-NN (Eğitilmiş: %{egitilmis_val[0]:.1f})", fontsize=12, fontweight="bold", color="#2c5282")
        ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 3: Few-Shot Veri Verimliliği Eğrisi
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        oranlar = few_shot_egrisi["oranlar"]
        dogruluklar = few_shot_egrisi["dogruluklar"]
        
        ax3.plot(oranlar, dogruluklar, "o-", color="#38a169", linewidth=2.5, markersize=7, label="Few-Shot Linear Probing")
        ax3.axhline(egitilmis_sonuclar["linear_probe_100"], color="#e53e3e", linestyle="--", label="Tam Veri (%100 Etiket)")
        
        for r, acc in zip(oranlar, dogruluklar):
            ax3.annotate(f"%{acc:.1f}", (r, acc), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold")

        ax3.set_title("3. Few-Shot Veri Verimliliği (%1, %10, %50, %100)", fontsize=12, fontweight="bold", color="#276749")
        ax3.set_xlabel("Kullanılan Etiket Oranı (%)", fontsize=10)
        ax3.set_ylabel("Doğruluk (%)", fontsize=10)
        ax3.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 4: Temsil Uzayı Manifold Ayrışması (PCA 2D)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        benzersiz_siniflar = np.unique(etiketler)
        palette = sns.color_palette("bright", len(benzersiz_siniflar))
        for idx, c in enumerate(benzersiz_siniflar):
            mask = (etiketler == c)
            ax4.scatter(
                gomulmeler_2d[mask, 0], gomulmeler_2d[mask, 1],
                color=palette[idx], label=f"Sınıf {int(c)}",
                alpha=0.85, s=45
            )
        ax4.set_title("4. Doğrusal Olarak Ayrışabilir Temsil Uzayı (PCA)", fontsize=12, fontweight="bold", color="#2c7a7b")
        ax4.set_xlabel("Boyut 1", fontsize=10)
        ax4.set_ylabel("Boyut 2", fontsize=10)
        ax4.legend(loc="upper right", fontsize=8, framealpha=0.8)
        ax4.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 5: Temsil Geometrisi ve Kalite Metrikleri
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        geo_isimler = ["Silhouette", "İzotropi", "Efektif Boyut", "Kosinüs Marjin"]
        geo_degerler = [
            egitilmis_sonuclar["silhouette_skoru"],
            egitilmis_sonuclar["izotropi_indeksi"],
            egitilmis_sonuclar["efektif_boyut"] / 64.0, # normalize
            egitilmis_sonuclar["ayrisma_marjini"]
        ]
        
        colors = ["#4299e1", "#ed8936", "#48bb78", "#9f7aea"]
        bars = ax5.bar(geo_isimler, geo_degerler, color=colors, width=0.5, alpha=0.85)
        for bar, val in zip(bars, geo_degerler):
            ax5.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

        ax5.set_ylim(0, 1.25)
        ax5.set_title(f"5. Manifold Kalite İndeksleri (Silhouette: {egitilmis_sonuclar['silhouette_skoru']:.3f})", fontsize=12, fontweight="bold", color="#4a5568")
        ax5.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 6: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        
        swot_metni = (
            "           TEMSİL BENCHMARK SUITE SWOT MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • Dondurulmuş omurga ile sıfır veri sızıntısı ve objektif kıyaslama.\n"
            "  • k-NN ile hiperparametre optimizasyonu olmadan saf manifold testi.\n"
            "  • Few-shot değerlendirmesi ile gerçek dünya veri verimliliği ölçümü.\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • Tüm veri kümesinin embedding çıkarımı için GPU bellek/zaman gerektirir.\n"
            "  • k-NN arama karmaşıklığı büyük doğrulama setlerinde O(N_val * N_train).\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • Foundation modellerin (DINO, CLIP, MAE) downstream başarısını tahmin etme.\n"
            "  • Model sürümleme ve CI/CD regresyon testlerine otomatik kalite kapısı.\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • Doğrulama veri kümesinin dar veya dengesiz olması durumunda sapma."
        )
        
        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f7fafc", edgecolor="#4a5568", linewidth=1.8)
        )
        ax6.set_title("6. Temsil Benchmark Mimarisi SWOT Matrisi", fontsize=12, fontweight="bold", color="#2d3748")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
