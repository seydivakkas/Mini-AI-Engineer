"""Makine Öğrenmesi Tabanlı Aykırı Değer Tespiti Modülü (İzolasyon Ormanı & LOF).

Bu modül; doğrusal olmayan, çok boyutlu ve değişken yoğunluklu veri dağılımlarında
geometrik ve yoğunluk tabanlı aykırı değerleri tespit etmek için
İzolasyon Ormanı (Isolation Forest) ve Lokal Aykırı Faktör (Local Outlier Factor) algoritmalarını sunar.
"""

from typing import Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class IzolasyonOrmaniTespitEdici:
    """Ağaç tabanlı rastgele uzay bölmeleme ile Aykırı Değer Tespiti (Isolation Forest)."""

    def __init__(
        self,
        kirlilik_orani: float = 0.05,
        agac_sayisi: int = 100,
        rastgele_durum: int = 42
    ) -> None:
        """İzolasyon Ormanı tespit edicisini başlatır.

        Parametreler:
            kirlilik_orani (float): Veri kümesinde beklenen aykırı değer yüzdesi (0.01 - 0.50).
            agac_sayisi (int): Ormandaki bağımsız karar ağacı adedi.
            rastgele_durum (int): Tekrarlanabilirlik için tohum değeri.
        """
        self.kirlilik_orani = float(kirlilik_orani)
        self.agac_sayisi = int(agac_sayisi)
        self.rastgele_durum = int(rastgele_durum)

        self._model = IsolationForest(
            contamination=self.kirlilik_orani,
            n_estimators=self.agac_sayisi,
            random_state=self.rastgele_durum,
            n_jobs=-1
        )
        self._egitildi_mi = False

    def egit_ve_tespit_et(self, veri: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Modeli eğitir ve aykırı noktaları tespit eder.

        Döndürür:
            np.ndarray: Aykırı noktalar için True, normal noktalar için False (bool dizisi).
        """
        X = np.asarray(veri, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        tahminler = self._model.fit_predict(X)
        self._egitildi_mi = True
        # scikit-learn -1'i aykırı (outlier), 1'i normal (inlier) olarak kodlar
        return tahminler == -1

    def anomali_skorlarini_al(self, veri: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Her bir örneğe ait anomali karar skorunu döndürür (Skor ne kadar düşükse o kadar aykırı)."""
        if not self._egitildi_mi:
            raise RuntimeError("Model henüz eğitilmedi. Önce egit_ve_tespit_et() çağrılmalıdır.")

        X = np.asarray(veri, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return self._model.decision_function(X)


class LokalAykiriFaktorTespitEdici:
    """k-En Yakın Komşuluk ve Yerel Yoğunluk Tabanlı Aykırı Değer Tespiti (Local Outlier Factor)."""

    def __init__(
        self,
        komsu_sayisi: int = 20,
        kirlilik_orani: float = 0.05
    ) -> None:
        """LOF tespit edicisini başlatır.

        Parametreler:
            komsu_sayisi (int): Yerel yoğunluğu hesaplamak için bakılacak k-komşu sayısı.
            kirlilik_orani (float): Veri kümesinde beklenen anomali oranı.
        """
        self.komsu_sayisi = int(komsu_sayisi)
        self.kirlilik_orani = float(kirlilik_orani)

        self._model = LocalOutlierFactor(
            n_neighbors=self.komsu_sayisi,
            contamination=self.kirlilik_orani,
            n_jobs=-1
        )

    def egit_ve_tespit_et(self, veri: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Verinin k-komşu yoğunluklarını çıkararak yerel aykırıları saptar.

        Döndürür:
            np.ndarray: Aykırı noktalar için True, normal noktalar için False.
        """
        X = np.asarray(veri, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        tahminler = self._model.fit_predict(X)
        return tahminler == -1

    def negatif_aykiri_faktoru(self) -> np.ndarray:
        """Eğitilmiş verideki her örneğin negatif aykırı faktörü (LOF skoru)."""
        return self._model.negative_outlier_factor_
