"""
Day 325: Brain-Computer Interface (BCI) & Riemannian Geometry on EEG
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; çok kanallı EEG zaman serilerini, SCM kovaryans matrislerini,
Riemann manifold teğet uzayı projeksiyonunu ve BCI motor imgelemi sınıflandırma panosunu barındırır.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class RiemannGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü Riemannian BCI Teşhis ve Performans Panosu.
    """
    def __init__(self, cikti_dizini: str = None):
        if cikti_dizini is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cikti_dizini = os.path.join(base_dir, "ciktilar")
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI", "Arial"]
        plt.rcParams["axes.edgecolor"] = "#2c3e50"
        plt.rcParams["axes.linewidth"] = 1.2

    def teshis_panelini_ciz(
        self,
        sample_eeg: np.ndarray,
        sample_scms: Dict[int, np.ndarray],
        tangent_features: np.ndarray,
        labels: np.ndarray,
        riemann_dist_matrix: np.ndarray,
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "riemann_bci_paneli.png"
    ) -> str:
        """
        6 Panelli Riemannian BCI Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.suptitle(
            "EEG Motor Imagery & Riemannian Geometry BCI Teşhis Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        channels, num_samples = sample_eeg.shape
        t = np.arange(num_samples) / 250.0

        # ------------------------------------------------------------------
        # Panel 1: Ham Çok Kanallı EEG Zaman Serisi Sinyalleri
        # ------------------------------------------------------------------
        ax1 = axes[0, 0]
        channel_names = ["Fz", "C3", "Cz", "C4", "Pz", "O1", "O2", "Oz"]
        for c in range(min(5, channels)):
            offset = c * 4.0
            ax1.plot(t, sample_eeg[c] + offset, label=channel_names[c] if c < len(channel_names) else f"Ch {c+1}", alpha=0.85)
        ax1.set_title("1. Ham Çok Kanallı EEG Motor Korteks Sinyalleri", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("Zaman (saniye)", fontsize=8)
        ax1.set_ylabel("Sinyal Genliği (uV + Offset)", fontsize=8)
        ax1.legend(loc="upper right", fontsize=7)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 2: Sınıflara Göre Kovaryans Matrisi Heatmap'leri
        # ------------------------------------------------------------------
        ax2 = axes[0, 1]
        scm_left = sample_scms.get(0, np.eye(channels))
        im2 = ax2.imshow(scm_left, cmap="coolwarm", aspect="auto")
        plt.colorbar(im2, ax=ax2, label="Kovaryans Sigma")
        ax2.set_title("2. Sample Covariance Matrix (SCM) Sol El", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("EEG Kanalı", fontsize=8)
        ax2.set_ylabel("EEG Kanalı", fontsize=8)

        # ------------------------------------------------------------------
        # Panel 3: Riemann Manifold Teğet Uzayı PCA Projeksiyonu
        # ------------------------------------------------------------------
        ax3 = axes[0, 2]
        pca = PCA(n_components=2)
        tangent_2d = pca.fit_transform(tangent_features)
        
        colors = ["#e74c3c", "#2980b9", "#27ae60"]
        class_labels = ["Sol El (Class 0)", "Sağ El (Class 1)", "Ayaklar (Class 2)"]
        for c in range(3):
            mask = (labels == c)
            ax3.scatter(tangent_2d[mask, 0], tangent_2d[mask, 1], color=colors[c], label=class_labels[c], s=35, alpha=0.8)
        ax3.set_title("3. Teğet Uzayı Manifold Projeksiyonu (2D PCA)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Teğet Bileşen 1", fontsize=8)
        ax3.set_ylabel("Teğet Bileşen 2", fontsize=8)
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 4: AIRM Riemann Mesafe Matrisi Heatmap'i
        # ------------------------------------------------------------------
        ax4 = axes[1, 0]
        im4 = ax4.imshow(riemann_dist_matrix, cmap="viridis", aspect="auto")
        plt.colorbar(im4, ax=ax4, label="Riemann Mesafesi delta_R")
        ax4.set_title("4. Çiftli Riemann Mesafe Matrisi", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_xlabel("Deney İndeksi", fontsize=8)
        ax4.set_ylabel("Deney İndeksi", fontsize=8)

        # ------------------------------------------------------------------
        # Panel 5: Sınıflandırma Yöntemleri Doğruluk Karşılaştırması
        # ------------------------------------------------------------------
        ax5 = axes[1, 1]
        methods = ["Riemann MDM", "Tangent Space + SVM", "Öklid Basit Mean"]
        accuracies = [
            profiler_metrics.get("mdm_acc", 92.0),
            profiler_metrics.get("tangent_svm_acc", 96.5),
            profiler_metrics.get("euclidean_acc", 65.0)
        ]
        bars = ax5.bar(methods, accuracies, color=["#8e44ad", "#27ae60", "#e67e22"], width=0.5, alpha=0.85)
        for bar in bars:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"%{yval:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax5.set_title("5. BCI Sınıflandırma Başarımı (MDM vs Tangent SVM)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_ylabel("Doğruluk (%)", fontsize=8)
        ax5.set_ylim(0, 115)
        ax5.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 6: Riemannian BCI Hazır Bulunurluk Skoru
        # ------------------------------------------------------------------
        ax6 = axes[1, 2]
        metrics_list = ["Manifold Uyum Skoru", "Teğet Boyut Boyu", "SPD Kararlılık Skoru", "BCI Canlı Çıkarım"]
        scores = [
            profiler_metrics.get("manifold_score", 95.0),
            profiler_metrics.get("tangent_dim_score", 100.0),
            profiler_metrics.get("spd_stability_score", 98.0),
            profiler_metrics.get("bci_readiness_score", 94.0)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#3498db", alpha=0.8)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. Riemannian BCI Sistem Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
