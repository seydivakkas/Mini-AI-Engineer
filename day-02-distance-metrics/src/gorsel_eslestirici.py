"""Görsel Benzerlik ve En Yakın Komşu Eşleştirme Motoru.

Bu modül, bir görsel veya embedding katalogundaki verilerle sorgu görseli
arasında en yakın komşu (k-NN) aramasını ve sıralı geri çağırmayı (Ranked Retrieval)
MesafeOlcer çekirdeğini kullanarak gerçekleştirir.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from src.mesafe_olcer import MesafeOlcer


@dataclass(frozen=True)
class EslestirmeSonucu:
    """Tekil bir eşleşme öğesinin sıralama ve skor bilgisi."""

    sira: int
    oge_kimligi: str
    skor: float
    metrik: str


class GorselBenzerlikEslestirici:
    """Katalog tabanlı görsel ve vektör benzerlik arama motoru."""

    def __init__(self, metrik: str = "kosinus") -> None:
        """Eşleştiriciyi belirtilen metrik ile başlatır.

        Parametreler:
            metrik (str): 'kosinus', 'oklid' veya 'manhattan'.
        """
        gecerli_metrikler = ("kosinus", "oklid", "manhattan")
        if metrik not in gecerli_metrikler:
            raise ValueError(f"Geçersiz metrik: {metrik}. Seçenekler: {gecerli_metrikler}")

        self.metrik = metrik
        self._katalog_kimlikleri: List[str] = []
        self._katalog_vektorleri: Optional[np.ndarray] = None

    def katalog_ekle(self, kimlik: str, ozellik_vektoru: np.ndarray) -> None:
        """Kataloga yeni bir görsel öznitelik veya piksel vektörü ekler."""
        if not isinstance(ozellik_vektoru, np.ndarray):
            raise TypeError("Özellik vektörü NumPy ndarray olmalıdır.")

        vektor = ozellik_vektoru.astype(np.float64).ravel()

        if self._katalog_vektorleri is None:
            self._katalog_vektorleri = np.expand_dims(vektor, axis=0)
        else:
            if vektor.shape[0] != self._katalog_vektorleri.shape[1]:
                raise ValueError(
                    f"Vektör boyutu uyuşmuyor: {vektor.shape[0]} != {self._katalog_vektorleri.shape[1]}"
                )
            self._katalog_vektorleri = np.vstack([self._katalog_vektorleri, vektor])

        self._katalog_kimlikleri.append(kimlik)

    def en_yakin_k_bul(
        self,
        sorgu_vektoru: np.ndarray,
        k: int = 3
    ) -> List[EslestirmeSonucu]:
        """Sorgu vektörüne en çok benzeyen ilk K adet katalog öğesini getirir.

        Parametreler:
            sorgu_vektoru (np.ndarray): Sorgulanacak görsel öznitelik vektörü.
            k (int): Getirilecek sonuç sayısı.

        Döndürür:
            List[EslestirmeSonucu]: Sıralı eşleşme listesi.
        """
        if self._katalog_vektorleri is None or len(self._katalog_kimlikleri) == 0:
            raise RuntimeError("Katalog henüz boş, önce veri ekleyiniz.")

        k_sayisi = min(k, len(self._katalog_kimlikleri))
        sorgu = sorgu_vektoru.astype(np.float64).ravel()

        if self.metrik == "kosinus":
            skorlar = MesafeOlcer.toplu_kosinus_benzerligi(sorgu, self._katalog_vektorleri)
            # Kosinüs benzerliğinde yüksek skor daha iyidir (azalan sıralama)
            sirali_indeksler = np.argsort(skorlar)[::-1][:k_sayisi]
        elif self.metrik == "oklid":
            skorlar = MesafeOlcer.toplu_oklid_mesafesi(sorgu, self._katalog_vektorleri)
            # Mesafelerde düşük skor daha iyidir (artan sıralama)
            sirali_indeksler = np.argsort(skorlar)[:k_sayisi]
        elif self.metrik == "manhattan":
            # Broadcasting ile Manhattan mesafesi
            farklar = np.abs(self._katalog_vektorleri - sorgu)
            skorlar = np.sum(farklar, axis=1)
            sirali_indeksler = np.argsort(skorlar)[:k_sayisi]
        else:
            raise ValueError(f"Bilinmeyen metrik: {self.metrik}")

        sonuclar: List[EslestirmeSonucu] = []
        for sira_no, indeks in enumerate(sirali_indeksler, start=1):
            sonuclar.append(
                EslestirmeSonucu(
                    sira=sira_no,
                    oge_kimligi=self._katalog_kimlikleri[indeks],
                    skor=float(skorlar[indeks]),
                    metrik=self.metrik
                )
            )

        return sonuclar
