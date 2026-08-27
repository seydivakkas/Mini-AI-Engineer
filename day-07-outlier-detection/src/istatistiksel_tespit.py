"""İstatistiksel Aykırı Değer Tespiti Modülü (Z-Skoru, Modifiye Z & IQR).

Bu modül; tek değişkenli ve çok boyutlu eksenlerde istatistiksel parametrelere
(ortalama, standart sapma, medyan, MAD ve çeyrekler açıklığı) dayalı
hızlı ve matematiksel aykırı değer tespiti yapar.
"""

from typing import Tuple, Union
import numpy as np
import pandas as pd


class ZSkoruTespitEdici:
    """Klasik ve Modifiye (MAD Tabanlı) Z-Skoru ile Aykırı Değer Tespit Edici."""

    def __init__(
        self,
        esik_degeri: float = 3.0,
        modifiye_kullan: bool = False,
        epsilon: float = 1e-9
    ) -> None:
        """Tespit ediciyi yapılandırır.

        Parametreler:
            esik_degeri (float): Aykırı değer olarak etiketlenecek mutlak Z sınırı.
            modifiye_kullan (bool): True ise ortalama/std yerine Medyan ve MAD kullanır.
            epsilon (float): Sıfıra bölmeyi engelleyen sayısal dengeleyici.
        """
        if esik_degeri <= 0:
            raise ValueError("Eşik değeri pozitif bir sayı olmalıdır.")

        self.esik_degeri = float(esik_degeri)
        self.modifiye_kullan = bool(modifiye_kullan)
        self.epsilon = float(epsilon)

    def skorlari_hesapla(self, veri: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Veri için tek tek Z-Skoru veya Modifiye Z-Skoru üretir."""
        dizi = np.asarray(veri, dtype=np.float64)

        if not self.modifiye_kullan:
            ortalama = np.mean(dizi)
            std_sapma = np.std(dizi)
            z_skorlari = (dizi - ortalama) / (std_sapma + self.epsilon)
            return z_skorlari
        else:
            # Boris Iglewicz & David Hoaglin Modifiye Z formülü
            medyan = np.median(dizi)
            mutlak_sapmalar = np.abs(dizi - medyan)
            mad = np.median(mutlak_sapmalar)
            modifiye_z = (0.6745 * (dizi - medyan)) / (mad + self.epsilon)
            return modifiye_z

    def tespit_et(self, veri: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Aykırı değerleri tespit eder. Aykırı olan indekslerde True döner."""
        skorlar = self.skorlari_hesapla(veri)
        return np.abs(skorlar) > self.esik_degeri


class IqrAykiriDegerTespitEdici:
    """Tukey Çeyrekler Açıklığı (IQR) Yöntemiyle Aykırı Değer Tespit Edici."""

    def __init__(self, carpan: float = 1.5) -> None:
        """IQR tespit ediciyi yapılandırır.

        Parametreler:
            carpan (float): IQR katsayısı (1.5 hafif aykırı, 3.0 aşırı aykırı).
        """
        if carpan <= 0:
            raise ValueError("IQR çarpanı pozitif bir değer olmalıdır.")

        self.carpan = float(carpan)
        self.alt_sinir: float = 0.0
        self.ust_sinir: float = 0.0

    def sinirlari_ogren(self, veri: Union[np.ndarray, pd.Series]) -> Tuple[float, float]:
        """Q1, Q3 ve IQR sınırlarını hesaplar ve saklar."""
        dizi = np.asarray(veri, dtype=np.float64)
        q1 = float(np.percentile(dizi, 25))
        q3 = float(np.percentile(dizi, 75))
        iqr = q3 - q1

        self.alt_sinir = q1 - (self.carpan * iqr)
        self.ust_sinir = q3 + (self.carpan * iqr)
        return self.alt_sinir, self.ust_sinir

    def tespit_et(self, veri: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Hesaplanan sınırların dışına taşan noktaları True (aykırı) olarak işaretler."""
        dizi = np.asarray(veri, dtype=np.float64)
        self.sinirlari_ogren(dizi)
        return (dizi < self.alt_sinir) | (dizi > self.ust_sinir)
