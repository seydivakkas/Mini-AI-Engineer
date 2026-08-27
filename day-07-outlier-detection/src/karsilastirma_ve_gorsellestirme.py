"""Aykırı Değer Yöntemlerini Karşılaştırma ve Görselleştirme Modülü.

Bu modül; Z-Skoru, IQR, İzolasyon Ormanı ve LOF algoritmalarının tespitlerini
aynı veri üzerinde çalıştırarak mutabakat (consensus) puanlaması yapar ve
2x2 karşılaştırmalı grafik paneli üreterek diske kaydeder.
"""

from pathlib import Path
from typing import Any, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.istatistiksel_tespit import ZSkoruTespitEdici, IqrAykiriDegerTespitEdici
from src.makine_ogrenmesi_tespiti import IzolasyonOrmaniTespitEdici, LokalAykiriFaktorTespitEdici


class AykiriDegerKarsilastirici:
    """Tüm aykırı değer yöntemlerini tek çatı altında karşılaştıran ve oylayan motor."""

    def __init__(self, veri: np.ndarray) -> None:
        """Karşılaştırıcıyı başlatır.

        Parametreler:
            veri (np.ndarray): N x D boyutlu sayısal matris.
        """
        self.veri = np.asarray(veri, dtype=np.float64)
        if self.veri.ndim == 1:
            self.veri = self.veri.reshape(-1, 1)

        self.ornek_sayisi = self.veri.shape[0]

    def tum_yontemleri_calistir(
        self,
        kirlilik_orani: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """4 temel yöntemi çalıştırarak boolean maskeleri döndürür."""
        # Tek boyutlu istatistikler için verinin normu veya birinci ekseni kullanılır
        tek_boyut = np.linalg.norm(self.veri, axis=1) if self.veri.shape[1] > 1 else self.veri[:, 0]

        # 1. Z-Skoru
        z_bulucu = ZSkoruTespitEdici(esik_degeri=3.0)
        maske_z = z_bulucu.tespit_et(tek_boyut)

        # 2. IQR
        iqr_bulucu = IqrAykiriDegerTespitEdici(carpan=1.5)
        maske_iqr = iqr_bulucu.tespit_et(tek_boyut)

        # 3. İzolasyon Ormanı
        iso_bulucu = IzolasyonOrmaniTespitEdici(kirlilik_orani=kirlilik_orani)
        maske_iso = iso_bulucu.egit_ve_tespit_et(self.veri)

        # 4. LOF
        lof_bulucu = LokalAykiriFaktorTespitEdici(kirlilik_orani=kirlilik_orani)
        maske_lof = lof_bulucu.egit_ve_tespit_et(self.veri)

        return {
            "Z-Skoru": maske_z,
            "IQR (Tukey)": maske_iqr,
            "İzolasyon Ormanı": maske_iso,
            "Lokal Aykırı Faktör (LOF)": maske_lof,
        }

    def mutabakat_analizi(self, sonuclar: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Modellerin ortak oylarını (ensemble consensus) hesaplar."""
        oy_matrisi = np.column_stack(list(sonuclar.values()))
        toplam_oylar = np.sum(oy_matrisi, axis=1)  # 0 ile 4 arasında oy

        oy_oranlari = {
            "4/4 Oy Birliği (Kesin Anomali)": int(np.sum(toplam_oylar == 4)),
            "3/4 Oy Çokluğu (Yüksek Şüphe)": int(np.sum(toplam_oylar == 3)),
            "2/4 Kısmi Ayrışma": int(np.sum(toplam_oylar == 2)),
            "1/4 Tek Yöntem Uyarısı": int(np.sum(toplam_oylar == 1)),
            "0/4 Temiz / Normal": int(np.sum(toplam_oylar == 0)),
        }
        return {
            "toplam_oylar": toplam_oylar,
            "oy_dagilimi": oy_oranlari
        }


class AykiriDegerGorsellestirici:
    """Aykırı değer haritalarını 2x2 grid halinde PNG olarak çizen görselleştirici."""

    @staticmethod
    def karsilastirma_grafigi_ciz(
        veri_2d: np.ndarray,
        sonuclar: Dict[str, np.ndarray],
        dosya_yolu: Path,
        x_etiketi: str = "Öznitelik X1",
        y_etiketi: str = "Öznitelik X2"
    ) -> Path:
        """4 algoritmanın tespitlerini 2 boyutlu saçılım üzerinde karşılaştırmalı çizer."""
        fig, eksenler = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
        eksenler_listesi = eksenler.ravel()

        renkler = {
            "Z-Skoru": "#e74c3c",
            "IQR (Tukey)": "#e67e22",
            "İzolasyon Ormanı": "#9b59b6",
            "Lokal Aykırı Faktör (LOF)": "#27ae60",
        }

        for idx, (yontem_adi, maske) in enumerate(sonuclar.items()):
            ax = eksenler_listesi[idx]
            normal_noktalar = veri_2d[~maske]
            aykiri_noktalar = veri_2d[maske]

            # Normal noktalar (Mavi / Şeffaf)
            ax.scatter(
                normal_noktalar[:, 0], normal_noktalar[:, 1],
                color="#2980b9", alpha=0.5, s=25, label=f"Normal ({len(normal_noktalar)})"
            )

            # Aykırı noktalar (Kırmızı / Vurgulu X)
            ax.scatter(
                aykiri_noktalar[:, 0], aykiri_noktalar[:, 1],
                color=renkler.get(yontem_adi, "#c0392b"), alpha=0.9, s=65,
                marker="x", linewidths=2, label=f"Aykırı ({len(aykiri_noktalar)})"
            )

            ax.set_title(f"{yontem_adi} Tespiti", fontsize=11, fontweight="bold")
            ax.set_xlabel(x_etiketi, fontsize=9)
            ax.set_ylabel(y_etiketi, fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right", fontsize=8)

        fig.suptitle("Aykırı Değer Tespit Yöntemlerinin Karşılaştırmalı Dağılımı", fontsize=14, fontweight="bold")
        fig.tight_layout()

        dosya_yolu.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dosya_yolu)
        plt.close(fig)
        return dosya_yolu
