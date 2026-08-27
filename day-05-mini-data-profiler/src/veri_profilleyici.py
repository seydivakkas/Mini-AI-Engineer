"""Mini Veri Profilleme Motoru - İstatistiksel Metaveri Çıkarıcı.

Bu modül; tabüler veri çerçevelerinin sütun kardinalitesini, dağılım istatistiklerini
(çarpıklık, basıklık, çeyreklikler), eksik veri kalıplarını ve anlamsal tiplerini
otomatik olarak analiz eden hafif (lightweight) ve hızlı bir profilleme motoru sunar.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class SutunProfili:
    """Tek bir sütuna ait istatistiksel ve anlamsal profil dökümü."""

    sutun_adi: str
    fiziksel_tip: str
    anlamsal_tip: str  # 'sayisal_surekli', 'sayisal_ayrik', 'kategorik', 'benzersiz_kimlik'
    toplam_kayit: int
    eksik_sayisi: int
    eksik_orani: float
    benzersiz_sayisi: int
    kardinalite_orani: float
    en_sik_degerler: List[Tuple[Any, int, float]]
    istatistikler: Dict[str, float]
    uyarilar: List[str]


@dataclass(frozen=True)
class VeriKumesiProfili:
    """Veri setinin bütününe ait genel boyut, bellek ve sütun profilleri özeti."""

    satir_sayisi: int
    sutun_sayisi: int
    toplam_hucre: int
    toplam_eksik_hucre: int
    genel_eksik_orani: float
    toplam_bellek_kb: float
    satir_basi_bayt: float
    sutunlar: Dict[str, SutunProfili]
    genel_uyarilar: List[str]


class MiniVeriProfilleyici:
    """Otomatik veri seti profilleme ve metaveri analiz motoru."""

    def __init__(
        self,
        yuksek_eksik_esigi: float = 0.20,
        yuksek_carpiklik_esigi: float = 1.5,
        kardinalite_ayrik_esigi: int = 20
    ) -> None:
        """Profilleyici parametrelerini yapılandırır.

        Parametreler:
            yuksek_eksik_esigi (float): Uyarı tetikleyecek eksik veri oranı (varsayılan: %20).
            yuksek_carpiklik_esigi (float): Uyarı tetikleyecek mutlak çarpıklık eşiği.
            kardinalite_ayrik_esigi (int): Sayısal veriyi ayrık/kategorik sayma üst sınırı.
        """
        self.eksik_esigi = yuksek_eksik_esigi
        self.carpiklik_esigi = yuksek_carpiklik_esigi
        self.ayrik_esigi = kardinalite_ayrik_esigi

    def profili_cikar(self, veri_cercevesi: pd.DataFrame) -> VeriKumesiProfili:
        """Verilen DataFrame için kapsamlı istatistiksel ve anlamsal profil üretir."""
        if not isinstance(veri_cercevesi, pd.DataFrame):
            raise TypeError("Girdi bir pandas DataFrame olmalıdır.")

        satir_sayisi, sutun_sayisi = veri_cercevesi.shape
        if satir_sayisi == 0:
            raise ValueError("Boş bir DataFrame profillenemez (satır sayısı: 0).")

        toplam_hucre = satir_sayisi * sutun_sayisi
        toplam_eksik = int(veri_cercevesi.isna().sum().sum())
        genel_eksik_orani = float(toplam_eksik / toplam_hucre) if toplam_hucre > 0 else 0.0

        bellek_bayt = float(veri_cercevesi.memory_usage(deep=True).sum())
        bellek_kb = round(bellek_bayt / 1024.0, 2)
        satir_basi_bayt = round(bellek_bayt / satir_sayisi, 2)

        sutun_profilleri: Dict[str, SutunProfili] = {}
        genel_uyarilar: List[str] = []

        if genel_eksik_orani > self.eksik_esigi:
            genel_uyarilar.append(f"Veri setinde kritik düzeyde eksik veri var: %{genel_eksik_orani * 100:.1f}")

        for sutun in veri_cercevesi.columns:
            sutun_profilleri[sutun] = self._sutun_profille(veri_cercevesi[sutun], satir_sayisi)

        return VeriKumesiProfili(
            satir_sayisi=satir_sayisi,
            sutun_sayisi=sutun_sayisi,
            toplam_hucre=toplam_hucre,
            toplam_eksik_hucre=toplam_eksik,
            genel_eksik_orani=round(genel_eksik_orani, 4),
            toplam_bellek_kb=bellek_kb,
            satir_basi_bayt=satir_basi_bayt,
            sutunlar=sutun_profilleri,
            genel_uyarilar=genel_uyarilar
        )

    def _sutun_profille(self, seri: pd.Series, toplam_satir: int) -> SutunProfili:
        """Tek bir sütunun ayrıntılı istatistiksel ve dağılım profilini çıkarır."""
        sutun_adi = str(seri.name)
        fiziksel_tip = str(seri.dtype)

        eksik_sayisi = int(seri.isna().sum())
        eksik_orani = round(float(eksik_sayisi / toplam_satir), 4)

        gecerli_seri = seri.dropna()
        benzersiz_sayisi = int(gecerli_seri.nunique())
        kardinalite_orani = round(float(benzersiz_sayisi / len(gecerli_seri)), 4) if len(gecerli_seri) > 0 else 0.0

        # Anlamsal tipin belirlenmesi
        anlamsal_tip = self._anlamsal_tip_belirle(
            fiziksel_tip=fiziksel_tip,
            benzersiz_sayisi=benzersiz_sayisi,
            kardinalite_orani=kardinalite_orani,
            toplam_gecerli=len(gecerli_seri)
        )

        # En sık geçen 3 değer (Frekans tablosu)
        en_sik: List[Tuple[Any, int, float]] = []
        if len(gecerli_seri) > 0:
            frekanslar = gecerli_seri.value_counts(ascending=False).head(3)
            for deger, adet in frekanslar.items():
                yuzde = round(float(adet / len(gecerli_seri)) * 100.0, 2)
                en_sik.append((deger, int(adet), yuzde))

        # İstatistiklerin hesaplanması
        istatistikler: Dict[str, float] = {}
        uyarilar: List[str] = []

        if np.issubdtype(seri.dtype, np.number) and len(gecerli_seri) > 1:
            dizi = gecerli_seri.astype(np.float64).to_numpy()
            q25, q50, q75 = np.percentile(dizi, [25, 50, 75])
            iqr = q75 - q25

            carpiklik = float(stats.skew(dizi, bias=False)) if np.std(dizi) > 0 else 0.0
            basiklik = float(stats.kurtosis(dizi, bias=False)) if np.std(dizi) > 0 else 0.0

            istatistikler = {
                "en_kucuk": round(float(np.min(dizi)), 3),
                "en_buyuk": round(float(np.max(dizi)), 3),
                "ortalama": round(float(np.mean(dizi)), 3),
                "standart_sapma": round(float(np.std(dizi)), 3),
                "medyan": round(float(q50), 3),
                "yuzdelik_25": round(float(q25), 3),
                "yuzdelik_75": round(float(q75), 3),
                "iqr": round(float(iqr), 3),
                "carpiklik": round(carpiklik, 3),
                "basiklik": round(basiklik, 3),
            }

            if abs(carpiklik) > self.carpiklik_esigi:
                yon = "sağa (pozitif)" if carpiklik > 0 else "sola (negatif)"
                uyarilar.append(f"Yüksek çarpıklık ({carpiklik:.2f}): Dağılım {yon} kuyruklu.")

        # Uyarı üretimleri
        if eksik_orani > self.eksik_esigi:
            uyarilar.append(f"Yüksek eksiklik: %{eksik_orani * 100:.1f} kayıp veri.")
        if benzersiz_sayisi == 1:
            uyarilar.append("Sıfır Varyans (Sabit Sütun): Tüm değerler özdeş!")
        if kardinalite_orani > 0.95 and anlamsal_tip != "sayisal_surekli":
            uyarilar.append("Aday Benzersiz Anahtar (High Cardinality ID).")

        return SutunProfili(
            sutun_adi=sutun_adi,
            fiziksel_tip=fiziksel_tip,
            anlamsal_tip=anlamsal_tip,
            toplam_kayit=toplam_satir,
            eksik_sayisi=eksik_sayisi,
            eksik_orani=eksik_orani,
            benzersiz_sayisi=benzersiz_sayisi,
            kardinalite_orani=kardinalite_orani,
            en_sik_degerler=en_sik,
            istatistikler=istatistikler,
            uyarilar=uyarilar
        )

    def _anlamsal_tip_belirle(
        self,
        fiziksel_tip: str,
        benzersiz_sayisi: int,
        kardinalite_orani: float,
        toplam_gecerli: int
    ) -> str:
        """Sütunun fiziksel tipinden ve değer dağılımından anlamsal tipini çıkarır."""
        # Kayan noktalı (float) sürekli sayılar benzersiz kimlik olamaz
        if "float" in fiziksel_tip:
            if benzersiz_sayisi <= self.ayrik_esigi:
                return "sayisal_ayrik"
            return "sayisal_surekli"

        if kardinalite_orani > 0.95 and toplam_gecerli > 30 and ("int" in fiziksel_tip or "object" in fiziksel_tip or "string" in fiziksel_tip):
            return "benzersiz_kimlik"

        if "int" in fiziksel_tip:
            if benzersiz_sayisi <= self.ayrik_esigi:
                return "sayisal_ayrik"
            return "sayisal_surekli"

        if "category" in fiziksel_tip:
            return "kategorik"

        if "object" in fiziksel_tip:
            if kardinalite_orani < 0.20:
                return "kategorik"
            return "serbest_metin"

        return "diger"
