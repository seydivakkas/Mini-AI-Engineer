"""Renk Tabanlı Katalog Arama ve Sıralama Motoru.

Bu modül; bir e-ticaret veya desen veri tabanındaki ürünleri algısal renk
paletlerine göre indeksler, kullanıcı sorgusuna göre tüm kataloğu tarar
ve en benzer Top-K ürünü benzerlik yüzdeleriyle birlikte sıralar.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from src.palet_eslestirici import PaletRengi, PaletBenzerlikMotoru


@dataclass
class KatalogUrunu:
    """Katalogda yer alan ürünün kimliği, görseli ve renk paleti."""

    urun_id: str
    ad: str
    kategori: str
    gorsel_bgr: np.ndarray
    palet: List[PaletRengi]


@dataclass
class AramaSonucu:
    """Tek bir ürün için arama benzerlik sonucu."""

    urun: KatalogUrunu
    benzerlik_skoru: float  # [0.0, 100.0]
    delta_e_mesafesi: float
    eslesmeler: List[Dict[str, Any]]


class RenkTabanliAramaMotoru:
    """Ürün paletlerini indeksleyen ve sorgu paletine göre sıralayan motor."""

    def __init__(self, metrik: str = "ciede2000", hassasiyet_sigma: float = 25.0) -> None:
        self.katalog: List[KatalogUrunu] = []
        self.benzerlik_motoru = PaletBenzerlikMotoru(metrik=metrik, hassasiyet_sigma=hassasiyet_sigma)

    def urun_ekle(self, urun: KatalogUrunu) -> None:
        """Kataloğa yeni bir ürün ekler."""
        self.katalog.append(urun)

    def urunleri_toplu_ekle(self, urunler: List[KatalogUrunu]) -> None:
        """Kataloğa birden fazla ürünü topluca ekler."""
        self.katalog.extend(urunler)

    def arama_yap(
        self,
        sorgu_paleti: List[PaletRengi],
        en_iyi_k: int = 3
    ) -> List[AramaSonucu]:
        """Sorgu paletini tüm katalogdaki ürünlerle kıyaslayıp en benzer K ürünü döndürür."""
        if not self.katalog:
            raise ValueError("Katalog boş. Önce ürün ekleyiniz.")

        sonuclar: List[AramaSonucu] = []

        for urun in self.katalog:
            skor, mesafe, eslesmeler = self.benzerlik_motoru.benzerlik_skoru_hesapla(
                sorgu_paleti, urun.palet
            )
            sonuclar.append(
                AramaSonucu(
                    urun=urun,
                    benzerlik_skoru=skor,
                    delta_e_mesafesi=mesafe,
                    eslesmeler=eslesmeler
                )
            )

        # Benzerlik skoruna göre azalan sırada sırala
        sonuclar.sort(key=lambda s: s.benzerlik_skoru, reverse=True)
        return sonuclar[:en_iyi_k]
