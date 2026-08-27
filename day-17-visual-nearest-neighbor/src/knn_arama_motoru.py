"""Vektör Tabanlı Görsel Arama ve k-NN Eşleştirme Motoru.

Bu modül; görsel öznitelik vektörlerini indeksler, Öklid (L2), Kosinüs (Cosine)
ve Manhattan (L1) metrikleriyle Top-K en yakın görseli arar ve benzerlik skoru hesaplar.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
from src.vektor_cikarici import GorselVektorCikarici


@dataclass
class AramaSonucu:
    """Bir arama eşleşmesinin detaylı sonuç verisi."""

    sira: int
    indeks: int
    etiket: str
    mesafe: float
    benzerlik_yuzdesi: float
    gorsel_bgr: np.ndarray


class GorselAramaMotoru:
    """Yüksek boyutlu görsel gömmeleri indeksleyen ve k-NN sorguları çalıştıran motor."""

    def __init__(
        self,
        vektor_cikarici: Optional[GorselVektorCikarici] = None,
        varsayilan_metrik: str = "cosine"
    ) -> None:
        self.cikarici = vektor_cikarici or GorselVektorCikarici()
        self.varsayilan_metrik = varsayilan_metrik.lower()
        self.katalog_etiketler: List[str] = []
        self.katalog_gorseller: List[np.ndarray] = []
        self.katalog_matrisi: Optional[np.ndarray] = None  # (N, D) float32

    def katalog_ekle(
        self,
        etiket: str,
        gorsel_bgr: np.ndarray,
        vektor: Optional[np.ndarray] = None
    ) -> None:
        """Kataloğa tek bir görsel ekler."""
        if vektor is None:
            vektor = self.cikarici.vektor_cikar(gorsel_bgr)

        self.katalog_etiketler.append(etiket)
        self.katalog_gorseller.append(gorsel_bgr)

        vektor_2d = np.expand_dims(vektor.astype(np.float32), axis=0)
        if self.katalog_matrisi is None:
            self.katalog_matrisi = vektor_2d
        else:
            self.katalog_matrisi = np.vstack([self.katalog_matrisi, vektor_2d])

    def katalog_toplu_indeksle(self, gorsel_haritasi: Dict[str, np.ndarray]) -> int:
        """Sözlük biçiminde verilen etiket ve görselleri toplu olarak indeksler."""
        for etiket, gorsel in gorsel_haritasi.items():
            self.katalog_ekle(etiket, gorsel)
        return len(self.katalog_etiketler)

    def en_yakin_k_ara(
        self,
        sorgu_gorseli_bgr: np.ndarray,
        k: int = 5,
        metrik: Optional[str] = None
    ) -> List[AramaSonucu]:
        """Sorgu görseline en yakın K adet katalog görselini bulur."""
        if self.katalog_matrisi is None or len(self.katalog_etiketler) == 0:
            raise ValueError("Arama yapabilmek için katalogda en az bir görsel indekslenmiş olmalıdır.")

        if k <= 0:
            raise ValueError(f"k değeri pozitif olmalıdır, verilen: {k}")

        k = min(k, len(self.katalog_etiketler))
        kullanilan_metrik = (metrik or self.varsayilan_metrik).lower()

        # Sorgu görselinin vektörünü çıkar
        sorgu_vektoru = self.cikarici.vektor_cikar(sorgu_gorseli_bgr)  # (D,)

        # Mesafe Hesaplama (Vektörize Matris İşlemleri)
        if kullanilan_metrik == "cosine":
            # Kosinüs Benzerliği: S = (u . v) / (|u| * |v|)
            # Vektörler zaten L2 normalize olduğu için sadece nokta çarpımı (Dot Product)
            nokta_carpim = np.dot(self.katalog_matrisi, sorgu_vektoru)
            nokta_carpim = np.clip(nokta_carpim, -1.0, 1.0)
            mesafeler = 1.0 - nokta_carpim
            benzerlikler = np.clip(nokta_carpim * 100.0, 0.0, 100.0)

        elif kullanilan_metrik == "l2":
            # Öklid Mesafesi: sqrt(sum((u - v)^2))
            fark = self.katalog_matrisi - sorgu_vektoru
            mesafeler = np.linalg.norm(fark, axis=1)
            # Birim kürede maksimum L2 mesafesi 2.0'dır
            benzerlikler = np.clip((1.0 - (mesafeler / 2.0)) * 100.0, 0.0, 100.0)

        elif kullanilan_metrik == "manhattan":
            fark = np.abs(self.katalog_matrisi - sorgu_vektoru)
            mesafeler = np.sum(fark, axis=1)
            maks_l1 = np.max(mesafeler) if np.max(mesafeler) > 0 else 1.0
            benzerlikler = np.clip((1.0 - (mesafeler / maks_l1)) * 100.0, 0.0, 100.0)

        else:
            raise ValueError(f"Desteklenmeyen mesafe metriği: {kullanilan_metrik}. ('cosine', 'l2', 'manhattan')")

        # Mesafeye göre küçükten büyüğe sırala (Top-K)
        sirali_indeksler = np.argsort(mesafeler)[:k]

        sonuclar: List[AramaSonucu] = []
        for sira_no, idx in enumerate(sirali_indeksler, start=1):
            sonuclar.append(AramaSonucu(
                sira=sira_no,
                indeks=int(idx),
                etiket=self.katalog_etiketler[idx],
                mesafe=float(mesafeler[idx]),
                benzerlik_yuzdesi=float(benzerlikler[idx]),
                gorsel_bgr=self.katalog_gorseller[idx]
            ))

        return sonuclar
