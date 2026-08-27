"""EDA Grafik ve Görselleştirme Üreteci (Headless Matplotlib).

Bu modül; sunucularda veya terminal ortamlarında GUI penceresi açmadan
yüksek çözünürlüklü korelasyon ısı haritaları, dağılım histogramları ve saçılım grafikleri
üreterek doğrudan PNG dosyası olarak kaydeder.
"""

from pathlib import Path
from typing import List, Optional
import matplotlib
# Headless (penceresiz) arka uç seçimi
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class EdaGrafikUreteci:
    """Keşifçi veri analizi görselleştirmelerini disk üzerine oluşturan araç."""

    @staticmethod
    def korelasyon_isi_haritasi(
        korelasyon_matrisi: pd.DataFrame,
        dosya_yolu: Path
    ) -> Path:
        """Korelasyon matrisini anotasyonlu bir ısı haritası (Heatmap) olarak kaydeder."""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        veri = korelasyon_matrisi.to_numpy()
        sutunlar = list(korelasyon_matrisi.columns)

        resim = ax.imshow(veri, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        fig.colorbar(resim, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(np.arange(len(sutunlar)))
        ax.set_yticks(np.arange(len(sutunlar)))
        ax.set_xticklabels(sutunlar, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(sutunlar, fontsize=9)

        # Değerlerin kutucuk içine yazılması
        for i in range(len(sutunlar)):
            for j in range(len(sutunlar)):
                deger = veri[i, j]
                ax.text(
                    j, i, f"{deger:.2f}",
                    ha="center", va="center",
                    color="white" if abs(deger) > 0.5 else "black",
                    fontsize=8, fontweight="bold"
                )

        ax.set_title("Öznitelik Korelasyon Isı Haritası (Pearson)", fontsize=12, fontweight="bold", pad=12)
        fig.tight_layout()

        dosya_yolu.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dosya_yolu)
        plt.close(fig)
        return dosya_yolu

    @staticmethod
    def dagilim_histogramlari(
        veri: pd.DataFrame,
        sayisal_sutunlar: List[str],
        dosya_yolu: Path
    ) -> Path:
        """Sayısal özniteliklerin histogram ve frekans dağılımlarını birleştirilmiş çizelgede kaydeder."""
        adet = len(sayisal_sutunlar)
        satir = int(np.ceil(adet / 2))
        sutun = 2 if adet > 1 else 1

        fig, eksenler = plt.subplots(satir, sutun, figsize=(10, 3.5 * satir), dpi=150)
        eksenler_duz = np.array(eksenler).ravel() if adet > 1 else [eksenler]

        for idx, sutun_adi in enumerate(sayisal_sutunlar):
            ax = eksenler_duz[idx]
            degerler = veri[sutun_adi].dropna()
            ax.hist(degerler, bins=25, color="#2b5c8f", edgecolor="black", alpha=0.75)
            ax.set_title(f"{sutun_adi} Dağılımı", fontsize=10, fontweight="bold")
            ax.set_ylabel("Frekans", fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Artık kalan boş subplot eksenlerini kapat
        for idx in range(adet, len(eksenler_duz)):
            fig.delaxes(eksenler_duz[idx])

        fig.tight_layout()
        dosya_yolu.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dosya_yolu)
        plt.close(fig)
        return dosya_yolu

    @staticmethod
    def sacilim_grafigi(
        veri: pd.DataFrame,
        x_adi: str,
        y_adi: str,
        dosya_yolu: Path,
        renk_sutunu: Optional[str] = None
    ) -> Path:
        """İki değişken arasındaki iki boyutlu saçılımı (Scatter Plot) kaydeder."""
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

        if renk_sutunu and renk_sutunu in veri.columns:
            kategoriler = veri[renk_sutunu].unique()
            for kat in kategoriler:
                alt_kume = veri[veri[renk_sutunu] == kat]
                ax.scatter(alt_kume[x_adi], alt_kume[y_adi], label=str(kat), alpha=0.6, edgecolors="none")
            ax.legend(title=renk_sutunu, fontsize=8)
        else:
            ax.scatter(veri[x_adi], veri[y_adi], color="#e74c3c", alpha=0.6, edgecolors="none")

        ax.set_xlabel(x_adi, fontsize=10, fontweight="bold")
        ax.set_ylabel(y_adi, fontsize=10, fontweight="bold")
        ax.set_title(f"{x_adi} vs. {y_adi} Saçılım Analizi", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        dosya_yolu.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dosya_yolu)
        plt.close(fig)
        return dosya_yolu
