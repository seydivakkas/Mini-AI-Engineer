"""Kovaryans Analizi ve Mahalanobis Mesafesi Çekirdek Modülü.

Bu modül; çok değişkenli veri dağılımlarında korelasyon ve varyans etkilerini
dikkate alarak kovaryans matrisi hesaplama, sayısal kararlı ters alma (regülarizasyon)
ve Mahalanobis mesafesini Öklid ile kıyaslamalı olarak vektörize biçimde sunar.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class KarsilastirmaSonucu:
    """Tek bir veri noktası için Öklid ve Mahalanobis mesafe kıyaslama çıktısı."""

    nokta_adi: str
    oklid_mesafesi: float
    mahalanobis_mesafesi: float
    oklid_sirasi: int
    mahalanobis_sirasi: int
    yorum: str


class KovaryansAnalizoru:
    """Çok değişkenli veri matrislerinin kovaryans ve korelasyon geometrisini çözen araç."""

    @staticmethod
    def kovaryans_matrisi_hesapla(
        veri_matrisi: np.ndarray,
        serbestlik_payi: int = 1
    ) -> np.ndarray:
        """NumPy dahili fonksiyonu kullanmadan saf lineer cebir ile kovaryans matrisi üretir.

        Formül:
            Sigma = (1 / (N - 1)) * (X - mu)^T * (X - mu)

        Parametreler:
            veri_matrisi (np.ndarray): (N, D) boyutunda veri matrisi (N: örneklem, D: değişken).
            serbestlik_payi (int): Örneklem varyansı için Bessel düzeltmesi (varsayılan: 1).

        Döndürür:
            np.ndarray: (D, D) boyutunda simetrik kovaryans matrisi (float64).
        """
        if not isinstance(veri_matrisi, np.ndarray):
            raise TypeError("Girdi bir NumPy ndarray olmalıdır.")

        if veri_matrisi.ndim != 2:
            raise ValueError(f"Veri matrisi 2B (N, D) olmalıdır. Alınan boyut: {veri_matrisi.shape}")

        orneklem_sayisi, degisken_sayisi = veri_matrisi.shape
        if orneklem_sayisi <= serbestlik_payi:
            raise ValueError(
                f"Kovaryans hesaplamak için örneklem sayısı ({orneklem_sayisi}) > serbestlik payı ({serbestlik_payi}) olmalıdır."
            )

        matris = veri_matrisi.astype(np.float64)
        ortalama_vektoru = np.mean(matris, axis=0)
        merkezlenmis_veri = matris - ortalama_vektoru

        kovaryans = np.dot(merkezlenmis_veri.T, merkezlenmis_veri) / (orneklem_sayisi - serbestlik_payi)
        return kovaryans

    @staticmethod
    def korelasyon_matrisi_hesapla(kovaryans_matrisi: np.ndarray) -> np.ndarray:
        """Kovaryans matrisini Pearson korelasyon matrisine (R) normalize eder.

        Formül:
            R_ij = Sigma_ij / sqrt(Sigma_ii * Sigma_jj)
        """
        varyanslar = np.diag(kovaryans_matrisi)
        if np.any(varyanslar <= 0):
            raise ValueError("Kovaryans köşegeni pozitif varyanslar içermelidir.")

        standart_sapmalar = np.sqrt(varyanslar)
        dis_carpim = np.outer(standart_sapmalar, standart_sapmalar)
        korelasyon = kovaryans_matrisi / dis_carpim
        return np.clip(korelasyon, -1.0, 1.0)

    @staticmethod
    def guvenli_kovaryans_tersi(
        kovaryans_matrisi: np.ndarray,
        duzenleme_katsayisi: float = 1e-6
    ) -> np.ndarray:
        """Tekil (singular) veya kötü koşullu matrisler için Tikhonov düzenlemeli tersini alır.

        Sigma_duzenli = Sigma + lambda * I
        """
        boyut = kovaryans_matrisi.shape[0]
        birim_matris = np.eye(boyut, dtype=np.float64)
        duzenlenmis_kovaryans = kovaryans_matrisi + (duzenleme_katsayisi * birim_matris)

        try:
            ters_matris = np.linalg.inv(duzenlenmis_kovaryans)
        except np.linalg.LinAlgError:
            # Matris yine de tekilse Moore-Penrose sözde-tersi (pseudo-inverse) kullanılır
            ters_matris = np.linalg.pinv(duzenlenmis_kovaryans)

        return ters_matris


class MahalanobisMesafeOlcer:
    """Bir referans dağılıma göre Mahalanobis ve Öklid mesafelerini ölçen model."""

    def __init__(
        self,
        referans_verisi: np.ndarray,
        duzenleme_katsayisi: float = 1e-6
    ) -> None:
        """Referans veri dağılımının merkezini ve kovaryans matrisinin tersini hazırlar.

        Parametreler:
            referans_verisi (np.ndarray): Dağılımı temsil eden (N, D) boyutunda matris.
            duzenleme_katsayisi (float): Tikhonov regülarizasyon katsayısı.
        """
        if not isinstance(referans_verisi, np.ndarray) or referans_verisi.ndim != 2:
            raise ValueError("Referans verisi (N, D) boyutunda 2B ndarray olmalıdır.")

        self._veri = referans_verisi.astype(np.float64)
        self.ortalama = np.mean(self._veri, axis=0)
        self.kovaryans = KovaryansAnalizoru.kovaryans_matrisi_hesapla(self._veri)
        self.korelasyon = KovaryansAnalizoru.korelasyon_matrisi_hesapla(self.kovaryans)
        self.ters_kovaryans = KovaryansAnalizoru.guvenli_kovaryans_tersi(
            self.kovaryans, duzenleme_katsayisi
        )

    def tekil_mahalanobis_mesafesi(self, nokta: np.ndarray) -> float:
        """Tek bir noktanın dağılım merkezine olan Mahalanobis mesafesini hesaplar.

        Formül:
            D_M(x, mu) = sqrt((x - mu)^T * Sigma^(-1) * (x - mu))
        """
        vektor = nokta.astype(np.float64).ravel()
        if vektor.shape[0] != self.ortalama.shape[0]:
            raise ValueError(
                f"Nokta boyutu uyuşmuyor: {vektor.shape[0]} != {self.ortalama.shape[0]}"
            )

        fark = vektor - self.ortalama
        kare_mesafe = float(np.dot(np.dot(fark, self.ters_kovaryans), fark))
        # Sayısal yuvarlamadan kaynaklı negatifliği önleme
        return float(np.sqrt(max(0.0, kare_mesafe)))

    def tekil_oklid_mesafesi(self, nokta: np.ndarray) -> float:
        """Noktanın dağılım merkezine olan standart Öklid mesafesini hesaplar.

        Formül:
            d_E(x, mu) = sqrt(sum((x - mu)^2))
        """
        vektor = nokta.astype(np.float64).ravel()
        fark = vektor - self.ortalama
        return float(np.sqrt(np.dot(fark, fark)))

    def toplu_mahalanobis_mesafesi(self, veri_matrisi: np.ndarray) -> np.ndarray:
        """(M, D) boyutundaki veri kümesinin tüm noktaları için Mahalanobis mesafelerini döngüsüz hesaplar.

        Vektörize formül:
            farklar = X - mu  -> (M, D)
            sol_carpim = farklar @ Sigma^(-1)  -> (M, D)
            mesafeler = sqrt(sum(sol_carpim * farklar, axis=1))  -> (M,)
        """
        matris = veri_matrisi.astype(np.float64)
        if matris.ndim == 1:
            matris = np.expand_dims(matris, axis=0)

        if matris.shape[1] != self.ortalama.shape[0]:
            raise ValueError(f"Sütun sayısı uyuşmuyor: {matris.shape[1]} != {self.ortalama.shape[0]}")

        farklar = matris - self.ortalama
        sol_carpim = np.dot(farklar, self.ters_kovaryans)
        kare_mesafeler = np.sum(sol_carpim * farklar, axis=1)
        return np.sqrt(np.maximum(0.0, kare_mesafeler))

    def kiyaslama_raporu_olustur(
        self,
        noktalar_sozlugu: Dict[str, np.ndarray]
    ) -> Dict[str, KarsilastirmaSonucu]:
        """Verilen noktalar için Öklid vs. Mahalanobis sıralama çelişkilerini analiz eder."""
        gecici_liste = []
        for isim, nokta in noktalar_sozlugu.items():
            oklid = self.tekil_oklid_mesafesi(nokta)
            mahalanobis = self.tekil_mahalanobis_mesafesi(nokta)
            gecici_liste.append({
                "isim": isim,
                "oklid": oklid,
                "mahalanobis": mahalanobis
            })

        # Sıralamaları belirleme
        gecici_liste.sort(key=lambda x: x["oklid"])
        for sira, oge in enumerate(gecici_liste, start=1):
            oge["oklid_sirasi"] = sira

        gecici_liste.sort(key=lambda x: x["mahalanobis"])
        for sira, oge in enumerate(gecici_liste, start=1):
            oge["mahalanobis_sirasi"] = sira

        sonuclar: Dict[str, KarsilastirmaSonucu] = {}
        for oge in gecici_liste:
            fark_sirasi = abs(oge["oklid_sirasi"] - oge["mahalanobis_sirasi"])
            if fark_sirasi > 0:
                yorum = f"Korelasyon etkisi! Öklid #{oge['oklid_sirasi']} görürken, Mahalanobis #{oge['mahalanobis_sirasi']} görüyor."
            else:
                yorum = "İki metrik de uyumlu."

            sonuclar[oge["isim"]] = KarsilastirmaSonucu(
                nokta_adi=oge["isim"],
                oklid_mesafesi=round(oge["oklid"], 4),
                mahalanobis_mesafesi=round(oge["mahalanobis"], 4),
                oklid_sirasi=oge["oklid_sirasi"],
                mahalanobis_sirasi=oge["mahalanobis_sirasi"],
                yorum=yorum
            )

        return sonuclar
