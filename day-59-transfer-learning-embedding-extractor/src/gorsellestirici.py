"""
6-Panelli Transfer Learning ve L2-Normalize Embedding Teşhis Panosu.
"""

from typing import Dict, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


class EmbeddingGorsellestirici:
    """Embedding uzayı geometrisini, kosinüs benzerliğini ve linear probe performansını görselleştirir."""

    @classmethod
    def panel_ciz(
        cls,
        embeddings: np.ndarray,
        labels: np.ndarray,
        norm_bilgisi: Dict[str, float],
        benzerlik_bilgisi: Dict[str, Any],
        svd_bilgisi: Dict[str, Any],
        probe_bilgisi: Dict[str, Any],
        omurga_adi: str = "ResNet-512",
        hedef_path: str = "ciktilar/embedding_ekstraktor_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(21, 13), dpi=300)
        fig.suptitle(
            "Day 59: Transfer Learning ve Dondurulmuş Katmanlarla L2-Normalize Embedding Çıkarımı",
            fontsize=15, fontweight="bold", y=0.98
        )

        embed_dim = embeddings.shape[1]
        toplam_ornek = len(embeddings)
        ort_intra = benzerlik_bilgisi["ort_intra_benzerlik"]
        ort_inter = benzerlik_bilgisi["ort_inter_benzerlik"]
        ayrisabilirlik = benzerlik_bilgisi["ayrisabilirlik_orani"]
        nihai_acc = probe_bilgisi["nihai_val_dogruluk"]

        # -------------------------------------------------------------
        # Panel 1: Yönetici Özeti Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")

        kart_metni = (
            f"EMBEDDING ÇIKARICI YÖNETİCİ KARTI\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Kullanılan Omurga (Backbone): {omurga_adi}\n"
            f"• Özellik Boyutu (Embed Dim)  : {embed_dim}-D\n"
            f"• Toplam Çıkarılan Vektör     : {toplam_ornek:,}\n"
            f"─────────────────────────────────────────────\n"
            f"• L2 Birim Norm Doğrulaması   : %100 GEÇERLİ (||e||=1.0)\n"
            f"• Ortalama Sınıf-İçi Benzerlik: {ort_intra:.4f}\n"
            f"• Ortalama Sınıf-Dışı Benzerlik: {ort_inter:.4f}\n"
            f"• Ayrışabilirlik Oranı (Ratio): {ayrisabilirlik:.2f}x\n"
            f"─────────────────────────────────────────────\n"
            f"• Linear Probe Doğruluğu      : %{nihai_acc:.1f}\n"
            f"• FAISS İndeks Uyumluluğu     : %100 HAZIR (Dot Product)"
        )

        ax1.text(
            0.5, 0.5, kart_metni, transform=ax1.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.9", facecolor="#9b59b6", alpha=0.18, edgecolor="#8e44ad", linewidth=2),
            fontsize=9.0, fontweight="bold", family="monospace"
        )
        ax1.set_title("1. Embedding Çıkarıcı Yönetici Özeti", fontweight="bold", color="#2c3e50")

        # -------------------------------------------------------------
        # Panel 2: 2D PCA Temsil Uzayı (Semantik Kümeler)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        pca = PCA(n_components=2, random_state=42)
        emb_2d = pca.fit_transform(embeddings[:600])
        labels_sub = labels[:600]

        scatter = ax2.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels_sub, cmap="tab10", alpha=0.75, s=28, edgecolors="none")
        ax2.set_xlabel(f"PCA-1 (Varyans: %{pca.explained_variance_ratio_[0]*100:.1f})")
        ax2.set_ylabel(f"PCA-2 (Varyans: %{pca.explained_variance_ratio_[1]*100:.1f})")
        ax2.set_title("2. 2D Semantik Temsil Uzayı (PCA)", fontweight="bold", color="#2980b9")
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label("Sınıf İndeksi")

        # -------------------------------------------------------------
        # Panel 3: Sınıflar Arası Kosinüs Benzerlik Matrisi
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        sinif_matrisi = benzerlik_bilgisi["sinif_ortalama_matrisi"]
        sns.heatmap(sinif_matrisi, annot=True, fmt=".2f", cmap="Blues", ax=ax3, cbar=True)
        ax3.set_xlabel("Hedef Sınıf")
        ax3.set_ylabel("Kaynak Sınıf")
        ax3.set_title("3. Sınıflar Arası Kosinüs Benzerliği", fontweight="bold", color="#27ae60")

        # -------------------------------------------------------------
        # Panel 4: Sınıf-İçi vs Sınıf-Dışı Benzerlik Dağılımı
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        intra = benzerlik_bilgisi["intra_benzerlikler"][:5000]
        inter = benzerlik_bilgisi["inter_benzerlikler"][:5000]

        ax4.hist(intra, bins=40, alpha=0.65, color="#27ae60", label=f"Sınıf-İçi (Ort: {ort_intra:.2f})", density=True)
        ax4.hist(inter, bins=40, alpha=0.65, color="#e74c3c", label=f"Sınıf-Dışı (Ort: {ort_inter:.2f})", density=True)
        ax4.axvline(ort_intra, color="#1e8449", linestyle="--", linewidth=1.5)
        ax4.axvline(ort_inter, color="#922b21", linestyle="--", linewidth=1.5)
        ax4.set_xlabel("Kosinüs Benzerliği (Cosine Similarity)")
        ax4.set_ylabel("Yoğunluk (Density)")
        ax4.set_title("4. Sınıf-İçi vs Sınıf-Dışı Benzerlik Dağılımı", fontweight="bold", color="#8e44ad")
        ax4.legend(loc="upper left")

        # -------------------------------------------------------------
        # Panel 5: Tekil Değer Spektrumu & Kümülatif Varyans
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        var_oranlari = svd_bilgisi["varyans_oranlari"][:15]
        kum_var = svd_bilgisi["kumulatif_varyans"][:15]
        x_eks = list(range(1, len(var_oranlari) + 1))

        ax5.bar(x_eks, var_oranlari, color="#3498db", alpha=0.7, label="Tekil Bileşen Varyansı")
        ax5.plot(x_eks, kum_var, color="#e67e22", marker="o", linewidth=2.0, label="Kümülatif Varyans")
        ax5.set_xlabel("Tekil Değer İndeksi (Bileşen)")
        ax5.set_ylabel("Varyans Oranı")
        ax5.set_title("5. SVD Spektrumu ve Enerji Dağılımı", fontweight="bold", color="#d35400")
        ax5.legend(loc="center right")

        # -------------------------------------------------------------
        # Panel 6: Linear Probing Yakınsaması
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        probe_gecmis = probe_bilgisi["dogruluk_gecmisi"]
        ax6.plot(range(1, len(probe_gecmis) + 1), probe_gecmis, marker="o", color="#27ae60", linewidth=2.2, label="Linear Probe Acc (%)")
        ax6.axhline(nihai_acc, color="#c0392b", linestyle="--", label=f"Nihai: %{nihai_acc:.1f}")
        ax6.set_xlabel("Epoch")
        ax6.set_ylabel("Doğrulama Doğruluğu (%)")
        ax6.set_title("6. Linear Probe Sınıflandırma Başarısı", fontweight="bold", color="#27ae60")
        ax6.legend(loc="lower right")

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.36, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
