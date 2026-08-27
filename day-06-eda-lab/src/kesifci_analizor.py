"""Kapsamlı Keşifçi Veri Analizi (EDA) Çekirdek Modülü.

Bu modül; doğrusal (Pearson) ve monotonik (Spearman) korelasyon analizini,
çoklu doğrusallık (Multicollinearity) tespiti için Varyans Şişme Faktörünü (VIF)
ve hedef değişken ilişkilerini saf lineer cebir ve istatistikle çözer.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VifSonucu:
    """Tek bir özniteliğe ait Varyans Şişme Faktörü (VIF) değerlendirmesi."""

    sutun_adi: str
    vif_degeri: float
    durum: str  # 'Dusuk Risk', 'Orta Risk', 'Kritik Coklu Dogrusallik'


@dataclass(frozen=True)
class KorelasyonRaporu:
    """Doğrusal ve sıra tabanlı korelasyon matrisleri ve kritik çiftler."""

    pearson_matrisi: pd.DataFrame
    spearman_matrisi: pd.DataFrame
    yuksek_korelasyonlu_ciftler: List[Tuple[str, str, float, str]]


@dataclass(frozen=True)
class HedefDegiskenRaporu:
    """Belirlenen hedef değişken (Target) ile diğer tüm özniteliklerin ilişki özeti."""

    hedef_sutun: str
    sayisal_korelasyonlar: Dict[str, float]
    kategorik_dagilimlar: Dict[str, Dict[str, float]]


class KesifciVeriAnalizoru:
    """Çok değişkenli keşifçi veri analizi ve istatistiksel ilişki motoru."""

    def __init__(self, veri_cercevesi: pd.DataFrame) -> None:
        """Analizörü başlatır ve doğrular.

        Parametreler:
            veri_cercevesi (pd.DataFrame): İncelenecek veri tablosu.
        """
        if not isinstance(veri_cercevesi, pd.DataFrame):
            raise TypeError("Girdi bir pandas DataFrame olmalıdır.")

        if veri_cercevesi.empty:
            raise ValueError("Boş bir DataFrame analiz edilemez.")

        self._veri = veri_cercevesi.copy()
        self.sayisal_sutunlar = list(self._veri.select_dtypes(include=[np.number]).columns)
        self.kategorik_sutunlar = list(self._veri.select_dtypes(exclude=[np.number]).columns)

    def korelasyon_analizi(self, esik_degeri: float = 0.70) -> KorelasyonRaporu:
        """Pearson ve Spearman korelasyonlarını hesaplar ve yüksek ilişkili çiftleri belirler.

        Parametreler:
            esik_degeri (float): Çoklu doğrusallık alarmı verecek mutlak korelasyon eşiği.

        Döndürür:
            KorelasyonRaporu: Matrisler ve kritik çiftlerin listesi.
        """
        if len(self.sayisal_sutunlar) < 2:
            raise ValueError("Korelasyon analizi için en az 2 sayısal sütun gereklidir.")

        sayisal_df = self._veri[self.sayisal_sutunlar].dropna()

        pearson = sayisal_df.corr(method="pearson")
        spearman = sayisal_df.corr(method="spearman")

        yuksek_ciftler: List[Tuple[str, str, float, str]] = []
        sutunlar = list(pearson.columns)

        for i in range(len(sutunlar)):
            for j in range(i + 1, len(sutunlar)):
                s1, s2 = sutunlar[i], sutunlar[j]
                r_pearson = float(pearson.loc[s1, s2])
                r_spearman = float(spearman.loc[s1, s2])

                if abs(r_pearson) >= esik_degeri or abs(r_spearman) >= esik_degeri:
                    not_bilgisi = (
                        "Doğrusal ve Monotonik Uyumlu"
                        if abs(r_pearson - r_spearman) < 0.15
                        else "Doğrusal Olmayan Monotonik Eğilim (Spearman Yüksek)"
                    )
                    yuksek_ciftler.append((s1, s2, round(r_pearson, 3), not_bilgisi))

        return KorelasyonRaporu(
            pearson_matrisi=pearson.round(3),
            spearman_matrisi=spearman.round(3),
            yuksek_korelasyonlu_ciftler=yuksek_ciftler
        )

    def vif_analizi(self, sabit_terim_ekle: bool = True) -> List[VifSonucu]:
        """Harici kütüphane gerektirmeden saf EKK (En Küçük Kareler) ile VIF hesaplar.

        Formül:
            VIF_i = 1 / (1 - R_i^2)

        Parametreler:
            sabit_terim_ekle (bool): Regresyona kesişim (intercept) katsayısı ekleme.

        Döndürür:
            List[VifSonucu]: Her sütun için hesaplanan VIF değerleri ve risk etiketleri.
        """
        if len(self.sayisal_sutunlar) < 2:
            return []

        df_temiz = self._veri[self.sayisal_sutunlar].dropna()
        if len(df_temiz) <= len(self.sayisal_sutunlar):
            raise ValueError("VIF hesaplaması için satır sayısı değişken sayısından fazla olmalıdır.")

        matris = df_temiz.to_numpy(dtype=np.float64)
        sonuclar: List[VifSonucu] = []

        for i, sutun_adi in enumerate(self.sayisal_sutunlar):
            hedef = matris[:, i]
            digerleri = np.delete(matris, i, axis=1)

            if sabit_terim_ekle:
                # Sabit terim (ones) ekleme
                kesisim = np.ones((digerleri.shape[0], 1), dtype=np.float64)
                X = np.hstack([kesisim, digerleri])
            else:
                X = digerleri

            # OLS Katsayıları: beta = (X^T X)^(-1) X^T y
            try:
                beta, kalintilar, _, _ = np.linalg.lstsq(X, hedef, rcond=None)
                tahmin = np.dot(X, beta)
                toplam_kare_farki = np.sum((hedef - np.mean(hedef)) ** 2)
                hata_kare_farki = np.sum((hedef - tahmin) ** 2)

                r_kare = 1.0 - (hata_kare_farki / (toplam_kare_farki + 1e-12))
                r_kare = np.clip(r_kare, 0.0, 0.999999)
                vif = float(1.0 / (1.0 - r_kare))
            except Exception:
                vif = float("inf")

            if vif > 10.0:
                durum = "Kritik Çoklu Doğrusallık (Modelden Çıkarılmalı!)"
            elif vif > 5.0:
                durum = "Orta Seviye Doğrusallık (Takip Edilmeli)"
            else:
                durum = "Düşük Risk (Güvenli)"

            sonuclar.append(
                VifSonucu(
                    sutun_adi=sutun_adi,
                    vif_degeri=round(vif, 2),
                    durum=durum
                )
            )

        return sonuclar

    def hedef_iliskisi_analizi(self, hedef_sutun: str) -> HedefDegiskenRaporu:
        """Seçilen hedef değişkenin diğer tüm sütunlarla ilişkisini modeller.

        Parametreler:
            hedef_sutun (str): İncelenecek bağımlı değişken (ör. 'kusurlu_mu').
        """
        if hedef_sutun not in self._veri.columns:
            raise KeyError(f"Hedef sütun bulunamadı: {hedef_sutun}")

        sayisal_korelasyonlar: Dict[str, float] = {}
        hedef_seri = self._veri[hedef_sutun]

        # Sayısal değişkenlerle korelasyon
        if np.issubdtype(hedef_seri.dtype, np.number):
            for s in self.sayisal_sutunlar:
                if s != hedef_sutun:
                    corr = float(self._veri[[s, hedef_sutun]].dropna().corr().iloc[0, 1])
                    sayisal_korelasyonlar[s] = round(corr, 3)

        # Kategorik değişkenlerle çapraz dağılım (Grup ortalamaları/frekansları)
        kategorik_oranlar: Dict[str, Dict[str, float]] = {}
        for k_sutun in self.kategorik_sutunlar:
            if k_sutun != hedef_sutun:
                if np.issubdtype(hedef_seri.dtype, np.number):
                    grup_ortalamasi = self._veri.groupby(k_sutun, observed=True)[hedef_sutun].mean()
                    kategorik_oranlar[k_sutun] = {
                        str(kat): round(float(val), 3) for kat, val in grup_ortalamasi.items()
                    }

        return HedefDegiskenRaporu(
            hedef_sutun=hedef_sutun,
            sayisal_korelasyonlar=sayisal_korelasyonlar,
            kategorik_dagilimlar=kategorik_oranlar
        )
