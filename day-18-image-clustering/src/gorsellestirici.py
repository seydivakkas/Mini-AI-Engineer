"""Kümeleme Görselleştirici Modülü.

Görsel embedding'lerinin PCA ile 2B izdüşümünü, Silhouette profilini,
optimal K analizini ve her kümeden temsili görsel galerisini
yüksek kaliteli bir grafik raporu olarak çizer ve kaydeder.
"""

from pathlib import Path
from typing import Dict, List, Optional
import cv2
import matplotlib
matplotlib.use("Agg")  # GUI gerektirmeyen arka uç
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from .kumeleme_motoru import KumelemeSonucu


class KumeGorsellestirici:
    """Kümeleme sonuçlarını ve metriklerini görselleştiren sınıf."""

    # Profesyonel renk paleti (Dark-mode uyumlu veya temiz açık tema)
    KUME_RENKLERI = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
    ]
    GURULTU_RENK = "#333333"

    @classmethod
    def kumeleme_raporu_olustur(
        cls,
        X: np.ndarray,
        gorseller: List[np.ndarray],
        kumeleme_sonucu: KumelemeSonucu,
        k_skorlari: Optional[Dict[int, float]] = None,
        hedef_dosya: Optional[Path] = None,
    ) -> Path:
        """Kapsamlı 4 panelli kümeleme görselleştirme raporu oluşturur.

        Paneller:
        1. Sol Üst: PCA 2B Küme İzdüşümü (Scatter Plot)
        2. Sağ Üst: K Değeri vs. Silhouette Skoru (Optimal K Analizi)
        3. Sol Alt: Örnek Bazlı Silhouette Profili (Silhouette Analysis)
        4. Sağ Alt: Kümelerden Temsili Görsel Galerisi (Image Grid Preview)

        Args:
            X: NxD boyutlu embedding matrisi.
            gorseller: N adet BGR formatında görsel.
            kumeleme_sonucu: Çalıştırılan algoritmanın KumelemeSonucu nesnesi.
            k_skorlari: K değerleri ve Silhouette skorları sözlüğü.
            hedef_dosya: Kaydedilecek PNG dosya yolu.

        Returns:
            Oluşturulan PNG raporunun dosya yolu.
        """
        if hedef_dosya is None:
            hedef_dosya = Path("ciktilar/kumeleme_raporu.png")
        hedef_dosya.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(18, 12), dpi=150)
        fig.patch.set_facecolor("#fafafa")

        etiketler = kumeleme_sonucu.etiketler
        benzersiz_etiketler = sorted(list(set(etiketler)))

        # ----------------------------------------------------
        # Panel 1: PCA 2B Projeksiyonu
        # ----------------------------------------------------
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_facecolor("#ffffff")
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)
        var_oran = pca.explained_variance_ratio_

        for etiket in benzersiz_etiketler:
            mask = etiketler == etiket
            if etiket == -1:
                renk = cls.GURULTU_RENK
                ad = f"Gürültü (Noise: {np.sum(mask)})"
                marker = "x"
                s = 80
            else:
                renk = cls.KUME_RENKLERI[etiket % len(cls.KUME_RENKLERI)]
                ad = f"Küme {etiket} (N={np.sum(mask)})"
                marker = "o"
                s = 120

            ax1.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=renk,
                label=ad,
                marker=marker,
                s=s,
                alpha=0.85,
                edgecolors="black" if marker == "o" else "red",
                linewidth=1.2,
            )

        ax1.set_title(
            f"PCA 2B Küme İzdüşümü — {kumeleme_sonucu.algoritma}\n"
            f"(Açıklanan Varyans: PC1=%{var_oran[0]*100:.1f}, PC2=%{var_oran[1]*100:.1f})",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax1.set_xlabel("Birinci Temel Bileşen (PC1)", fontsize=10)
        ax1.set_ylabel("İkinci Temel Bileşen (PC2)", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="best", frameon=True, facecolor="#f0f0f0", edgecolor="none")

        # ----------------------------------------------------
        # Panel 2: Optimal K vs. Silhouette Skoru
        # ----------------------------------------------------
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_facecolor("#ffffff")
        if k_skorlari:
            k_vals = list(k_skorlari.keys())
            sil_vals = list(k_skorlari.values())
            ax2.plot(
                k_vals,
                sil_vals,
                marker="s",
                markersize=8,
                linewidth=2.5,
                color="#e65100",
                label="Silhouette Skoru",
            )
            en_iyi_idx = np.argmax(sil_vals)
            ax2.scatter(
                [k_vals[en_iyi_idx]],
                [sil_vals[en_iyi_idx]],
                color="#b71c1c",
                s=180,
                zorder=5,
                label=f"Zirve: K={k_vals[en_iyi_idx]} (Skor={sil_vals[en_iyi_idx]:.3f})",
            )
            ax2.set_title(
                "Optimal K Taraması (Silhouette Analizi)",
                fontsize=12,
                fontweight="bold",
                pad=10,
            )
            ax2.set_xlabel("Küme Sayısı (K)", fontsize=10)
            ax2.set_ylabel("Ortalama Silhouette Katsayısı", fontsize=10)
            ax2.set_xticks(k_vals)
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend(loc="best", frameon=True, facecolor="#f0f0f0", edgecolor="none")
        else:
            ax2.text(
                0.5,
                0.5,
                "K Taraması Yapılmadı\n(DBSCAN veya Tekil Çalışma)",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12,
            )

        # ----------------------------------------------------
        # Panel 3: Silhouette Örnek Profili (Silhouette Analysis)
        # ----------------------------------------------------
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_facecolor("#ffffff")

        # Yalnızca geçerli kümeleri al (gürültüyü çıkar)
        gecerli_maske = etiketler != -1
        if len(set(etiketler[gecerli_maske])) >= 2:
            ornek_skorlari = silhouette_samples(
                X[gecerli_maske], etiketler[gecerli_maske], metric="cosine"
            )
            y_alt = 10
            ortalama_sil = np.mean(ornek_skorlari)

            for etiket in sorted(list(set(etiketler[gecerli_maske]))):
                kume_ornek_skorlari = ornek_skorlari[etiketler[gecerli_maske] == etiket]
                kume_ornek_skorlari.sort()
                kume_boyut = len(kume_ornek_skorlari)
                y_ust = y_alt + kume_boyut

                renk = cls.KUME_RENKLERI[etiket % len(cls.KUME_RENKLERI)]
                ax3.fill_betweenx(
                    np.arange(y_alt, y_ust),
                    0,
                    kume_ornek_skorlari,
                    facecolor=renk,
                    edgecolor="black",
                    alpha=0.75,
                )
                ax3.text(
                    -0.05,
                    y_alt + 0.5 * kume_boyut,
                    f"K {etiket}",
                    fontsize=10,
                    fontweight="bold",
                )
                y_alt = y_ust + 10

            ax3.axvline(
                x=ortalama_sil,
                color="#d32f2f",
                linestyle="--",
                linewidth=2,
                label=f"Ortalama Silhouette: {ortalama_sil:.3f}",
            )
            ax3.set_title(
                "Örnek Bazlı Silhouette Kalınlık & Ayrışma Profili",
                fontsize=12,
                fontweight="bold",
                pad=10,
            )
            ax3.set_xlabel("Silhouette Katsayısı s(i)", fontsize=10)
            ax3.set_ylabel("Örnek İndeksleri (Kümeler Halinde Gruplanmış)", fontsize=10)
            ax3.set_xlim([-0.2, 1.0])
            ax3.legend(loc="upper right", frameon=True, facecolor="#f0f0f0")
            ax3.grid(True, linestyle="--", alpha=0.5)
        else:
            ax3.text(
                0.5,
                0.5,
                "Silhouette Profili Hesaplanamadı\n(Yetersiz Küme Sayısı)",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12,
            )

        # ----------------------------------------------------
        # Panel 4: Kümelerden Temsili Görsel Galerisi (Mini Grid)
        # ----------------------------------------------------
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis("off")
        ax4.set_title(
            "Küme Temsilcileri Galerisi (Görsel Önizleme)",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        # Her kümeden ilk 3 görseli göster (maksimum 4 küme x 3 örnek = 12 küçük resim)
        satirlar = min(4, len(benzersiz_etiketler))
        sutunlar = 3
        alt_resim_boyut = 64
        tuval = np.full(
            (satirlar * (alt_resim_boyut + 20), sutunlar * (alt_resim_boyut + 20), 3),
            245,
            dtype=np.uint8,
        )

        for satir_idx, etiket in enumerate(benzersiz_etiketler[:satirlar]):
            ornek_indeksler = np.where(etiketler == etiket)[0][:sutunlar]
            for sutun_idx, img_idx in enumerate(ornek_indeksler):
                img = gorseller[img_idx]
                kucuk = cv2.resize(
                    img, (alt_resim_boyut, alt_resim_boyut), interpolation=cv2.INTER_AREA
                )
                
                y0 = satir_idx * (alt_resim_boyut + 20) + 10
                x0 = sutun_idx * (alt_resim_boyut + 20) + 10
                tuval[y0 : y0 + alt_resim_boyut, x0 : x0 + alt_resim_boyut] = kucuk

        tuval_rgb = cv2.cvtColor(tuval, cv2.COLOR_BGR2RGB)
        ax4.imshow(tuval_rgb)

        plt.suptitle(
            f"DAY 18: ETİKETSİZ GÖRSELLERİN OTOMATİK KÜMELENMESİ (IMAGE CLUSTERING)\n"
            f"Model: {kumeleme_sonucu.algoritma} | {kumeleme_sonucu.ozet()}",
            fontsize=14,
            fontweight="bold",
            color="#212121",
            y=0.98,
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(hedef_dosya, bbox_inches="tight", dpi=150)
        plt.close(fig)

        return hedef_dosya
