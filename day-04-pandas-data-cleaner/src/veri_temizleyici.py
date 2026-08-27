"""Pandas Tabanlı Üretim Seviyesi Tabüler Veri Temizleme ve Ön İşleme Motoru.

Bu modül; eksik veri doldurma (imputation), IQR tabanlı aykırı değer sınırlama (clipping),
mükerrer kayıt eleme ve bellek optimizasyonunu (downcasting) veri sızıntısını
(Data Leakage) engelleyen 'fit-transform' mimarisi ile gerçekleştirir.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemizlikRaporu:
    """Veri temizleme boru hattının çıktı ve optimizasyon metrikleri."""

    baslangic_satir_sayisi: int
    bitis_satir_sayisi: int
    elenen_yineleme_sayisi: int
    tamamlanan_eksik_degerler: Dict[str, int]
    budanan_aykiri_degerler: Dict[str, int]
    baslangic_bellek_kb: float
    bitis_bellek_kb: float
    bellek_kazanc_yuzdesi: float
    donusturulen_tipler: Dict[str, str]


class TabulerVeriTemizleyici:
    """Üretim seviyesinde tabüler veri temizleme ve bellek optimizasyon boru hattı."""

    def __init__(self) -> None:
        """Temizleyici sınıfını başlatır ve öğrenilecek parametre yapılarını kurar."""
        self._ogrenildi_mi: bool = False
        self._sayisal_tamamlama_degerleri: Dict[str, float] = {}
        self._kategorik_tamamlama_degerleri: Dict[str, Any] = {}
        self._aykiri_deger_sinirlari: Dict[str, Tuple[float, float]] = {}
        self._hedef_tipler: Dict[str, str] = {}

    @property
    def ogrenildi_mi(self) -> bool:
        """Modelin eğitim parametrelerini öğrenip öğrenmediğini döndürür."""
        return self._ogrenildi_mi

    def fit(
        self,
        veri_cercevesi: pd.DataFrame,
        sayisal_strateji: str = "medyan",
        kategorik_strateji: str = "mod",
        iqr_carpani: float = 1.5
    ) -> "TabulerVeriTemizleyici":
        """Eğitim verisinden istatistiksel parametreleri öğrenir (Veri sızıntısını önler).

        Parametreler:
            veri_cercevesi (pd.DataFrame): Parametrelerin çıkarılacağı eğitim tablosu.
            sayisal_strateji (str): 'medyan' veya 'ortalama'.
            kategorik_strateji (str): 'mod' veya 'bilinmiyor'.
            iqr_carpani (float): IQR aykırı değer eşik çarpanı (varsayılan: 1.5).

        Döndürür:
            TabulerVeriTemizleyici: Eğitilmiş temizleyici örneği (method chaining).
        """
        if not isinstance(veri_cercevesi, pd.DataFrame):
            raise TypeError("Girdi bir pandas DataFrame olmalıdır.")

        if veri_cercevesi.empty:
            raise ValueError("Eğitim için boş DataFrame verilemez.")

        self._sayisal_tamamlama_degerleri.clear()
        self._kategorik_tamamlama_degerleri.clear()
        self._aykiri_deger_sinirlari.clear()
        self._hedef_tipler.clear()

        sayisal_sutunlar = veri_cercevesi.select_dtypes(include=[np.number]).columns
        kategorik_sutunlar = veri_cercevesi.select_dtypes(exclude=[np.number]).columns

        # 1. Sayısal sütun parametrelerini öğrenme
        for sutun in sayisal_sutunlar:
            seri = veri_cercevesi[sutun].dropna()
            if seri.empty:
                continue

            # Eksik veri tamamlama değeri
            if sayisal_strateji == "medyan":
                self._sayisal_tamamlama_degerleri[sutun] = float(seri.median())
            elif sayisal_strateji == "ortalama":
                self._sayisal_tamamlama_degerleri[sutun] = float(seri.mean())
            else:
                raise ValueError(f"Geçersiz sayısal strateji: {sayisal_strateji}")

            # IQR Aykırı değer sınırlarını öğrenme: Q1, Q3
            q1 = float(seri.quantile(0.25))
            q3 = float(seri.quantile(0.75))
            iqr = q3 - q1
            alt_sinir = q1 - (iqr_carpani * iqr)
            ust_sinir = q3 + (iqr_carpani * iqr)
            self._aykiri_deger_sinirlari[sutun] = (alt_sinir, ust_sinir)

        # 2. Kategorik sütun parametrelerini öğrenme
        for sutun in kategorik_sutunlar:
            seri = veri_cercevesi[sutun].dropna()
            if seri.empty:
                self._kategorik_tamamlama_degerleri[sutun] = "Bilinmiyor"
            else:
                if kategorik_strateji == "mod":
                    mod_deger = seri.mode()
                    self._kategorik_tamamlama_degerleri[sutun] = mod_deger.iloc[0] if not mod_deger.empty else "Bilinmiyor"
                elif kategorik_strateji == "bilinmiyor":
                    self._kategorik_tamamlama_degerleri[sutun] = "Bilinmiyor"
                else:
                    raise ValueError(f"Geçersiz kategorik strateji: {kategorik_strateji}")

        self._ogrenildi_mi = True
        return self

    def transform(
        self,
        veri_cercevesi: pd.DataFrame,
        yinelemeleri_ele: bool = True,
        aykirilari_buda: bool = True,
        tipleri_optimize_et: bool = True,
        kopyala: bool = True
    ) -> pd.DataFrame:
        """Öğrenilen parametreleri uygulayarak veriyi temizler.

        Parametreler:
            veri_cercevesi (pd.DataFrame): Temizlenecek veri çerçevesi.
            yinelemeleri_ele (bool): True ise birebir kopya satırları siler.
            aykirilari_buda (bool): True ise IQR sınırlarının dışındaki değerleri sınırlar (clip).
            tipleri_optimize_et (bool): Bellek tasarrufu için sayısal ve kategorik tipleri küçültür.
            kopyala (bool): Orijinal veriyi korumak için derin kopya alır.

        Döndürür:
            pd.DataFrame: Temizlenmiş ve optimize edilmiş tablo.
        """
        if not self._ogrenildi_mi:
            raise RuntimeError("Temizleyici henüz eğitilmedi. Önce fit() çağrılmalıdır.")

        df = veri_cercevesi.copy() if kopyala else veri_cercevesi

        # 1. Adım: Duplikasyon (Mükerrer Kayıt) Eleme
        if yinelemeleri_ele:
            df = df.drop_duplicates(ignore_index=True)

        # 2. Adım: Eksik Veri Tamamlama (Imputation)
        for sutun, deger in self._sayisal_tamamlama_degerleri.items():
            if sutun in df.columns:
                df[sutun] = df[sutun].fillna(deger)

        for sutun, deger in self._kategorik_tamamlama_degerleri.items():
            if sutun in df.columns:
                df[sutun] = df[sutun].fillna(deger)

        # 3. Adım: IQR Aykırı Değer Kırpma (Winsorization / Clipping)
        if aykirilari_buda:
            for sutun, (alt, ust) in self._aykiri_deger_sinirlari.items():
                if sutun in df.columns:
                    df[sutun] = df[sutun].clip(lower=alt, upper=ust)

        # 4. Adım: Tip Dönüşümleri ve Bellek İndirgeme (Downcasting)
        if tipleri_optimize_et:
            df = self._bellek_optimize_et(df)

        return df

    def fit_transform(
        self,
        veri_cercevesi: pd.DataFrame,
        sayisal_strateji: str = "medyan",
        kategorik_strateji: str = "mod",
        iqr_carpani: float = 1.5,
        yinelemeleri_ele: bool = True,
        aykirilari_buda: bool = True,
        tipleri_optimize_et: bool = True
    ) -> pd.DataFrame:
        """Fit ve Transform adımlarını tek seferde icra eder."""
        return self.fit(
            veri_cercevesi,
            sayisal_strateji=sayisal_strateji,
            kategorik_strateji=kategorik_strateji,
            iqr_carpani=iqr_carpani
        ).transform(
            veri_cercevesi,
            yinelemeleri_ele=yinelemeleri_ele,
            aykirilari_buda=aykirilari_buda,
            tipleri_optimize_et=tipleri_optimize_et
        )

    def _bellek_optimize_et(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sayısal sütunları daha düşük bitli tiplere indirger, metinleri kategori yapar."""
        for sutun in df.columns:
            tur = df[sutun].dtype

            if np.issubdtype(tur, np.integer):
                df[sutun] = pd.to_numeric(df[sutun], downcast="integer")
            elif np.issubdtype(tur, np.floating):
                df[sutun] = pd.to_numeric(df[sutun], downcast="float")
            elif tur == "object":
                # Kardinalite (tekil değer oranı) %50'den az ise kategoriye çevrilir
                benzersiz_orani = df[sutun].nunique() / len(df)
                if benzersiz_orani < 0.50:
                    df[sutun] = df[sutun].astype("category")

        return df

    def rapor_olustur(
        self,
        ham_df: pd.DataFrame,
        temiz_df: pd.DataFrame
    ) -> TemizlikRaporu:
        """Ham ve temizlenmiş tabloları kıyaslayarak detaylı metrik raporu üretir."""
        baslangic_kb = ham_df.memory_usage(deep=True).sum() / 1024.0
        bitis_kb = temiz_df.memory_usage(deep=True).sum() / 1024.0
        kazanc = ((baslangic_kb - bitis_kb) / baslangic_kb) * 100.0 if baslangic_kb > 0 else 0.0

        tamamlananlar: Dict[str, int] = {}
        for sutun in ham_df.columns:
            if sutun in temiz_df.columns:
                eksik_farki = int(ham_df[sutun].isna().sum() - temiz_df[sutun].isna().sum())
                if eksik_farki > 0:
                    tamamlananlar[sutun] = eksik_farki

        budananlar: Dict[str, int] = {}
        for sutun, (alt, ust) in self._aykiri_deger_sinirlari.items():
            if sutun in ham_df.columns:
                seri = ham_df[sutun].dropna()
                aykiri_sayisi = int(((seri < alt) | (seri > ust)).sum())
                if aykiri_sayisi > 0:
                    budananlar[sutun] = aykiri_sayisi

        tipler: Dict[str, str] = {col: str(temiz_df[col].dtype) for col in temiz_df.columns}

        return TemizlikRaporu(
            baslangic_satir_sayisi=len(ham_df),
            bitis_satir_sayisi=len(temiz_df),
            elenen_yineleme_sayisi=len(ham_df) - len(temiz_df),
            tamamlanan_eksik_degerler=tamamlananlar,
            budanan_aykiri_degerler=budananlar,
            baslangic_bellek_kb=round(baslangic_kb, 2),
            bitis_bellek_kb=round(bitis_kb, 2),
            bellek_kazanc_yuzdesi=round(kazanc, 2),
            donusturulen_tipler=tipler
        )
