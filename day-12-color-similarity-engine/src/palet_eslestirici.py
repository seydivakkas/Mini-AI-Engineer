"""Çoklu Renk Paleti Eşleştirme ve Benzerlik Skorlama Motoru.

Bu modül; farklı sayıda ve farklı ağırlıklardaki iki renk paleti arasındaki
çift yönlü ağırlıklı algısal mesafeyi (Bidirectional Weighted Perceptual Distance)
ve %0-100 arası normalleştirilmiş benzerlik skorunu hesaplar.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
from src.delta_e_hesaplayici import DeltaEHesaplayici


@dataclass
class PaletRengi:
    """Tek bir palet rengini ve ağırlığını temsil eden veri sınıfı."""

    rgb: Tuple[int, int, int]
    agirlik: float  # [0.0, 1.0] aralığında yüzdesel ağırlık
    hex_kodu: str = ""
    lab: np.ndarray = None

    def __post_init__(self):
        if not self.hex_kodu:
            self.hex_kodu = f"#{self.rgb[0]:02X}{self.rgb[1]:02X}{self.rgb[2]:02X}"
        if self.lab is None:
            self.lab = DeltaEHesaplayici.rgb_to_lab(self.rgb)


class PaletBenzerlikMotoru:
    """İki renk paletini CIELAB uzayında kıyaslayan ve eşleştiren motor."""

    def __init__(self, metrik: str = "ciede2000", hassasiyet_sigma: float = 25.0) -> None:
        """Motoru yapılandırır.

        Parametreler:
            metrik (str): 'ciede2000' (en hassas) veya 'cie76' (hızlı).
            hassasiyet_sigma (float): Üstel benzerlik skoru ölçek parametresi.
        """
        self.metrik = metrik.lower()
        self.sigma = float(hassasiyet_sigma)

    def _iki_renk_mesafesi(self, lab1: np.ndarray, lab2: np.ndarray) -> float:
        """Seçilen metriğe göre iki renk arasındaki Delta-E farkını döndürür."""
        if self.metrik == "cie76":
            return DeltaEHesaplayici.cie76_mesafesi(lab1, lab2)
        return DeltaEHesaplayici.ciede2000_mesafesi(lab1, lab2)

    def palet_mesafesi_hesapla(
        self,
        palet_a: List[PaletRengi],
        palet_b: List[PaletRengi]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """İki palet arasındaki ağırlıklı algısal çift yönlü mesafeyi hesaplar.

        Döndürür:
            Tuple[float, List[Dict[str, Any]]]: (Toplam Ortalama Mesafe, Eşleşme Detayları)
        """
        if not palet_a or not palet_b:
            raise ValueError("Paletler boş olamaz.")

        # Ağırlıkların toplamını 1.0'a normalize et
        agirlik_a = np.array([r.agirlik for r in palet_a], dtype=np.float64)
        agirlik_b = np.array([r.agirlik for r in palet_b], dtype=np.float64)
        agirlik_a /= np.sum(agirlik_a)
        agirlik_b /= np.sum(agirlik_b)

        # 1. A'dan B'ye Ağırlıklı En Yakın Mesafe
        toplam_fark_a_to_b = 0.0
        eslesmeler_a = []
        for i, renk_a in enumerate(palet_a):
            en_kucuk_fark = float("inf")
            en_yakin_j = -1
            for j, renk_b in enumerate(palet_b):
                fark = self._iki_renk_mesafesi(renk_a.lab, renk_b.lab)
                if fark < en_kucuk_fark:
                    en_kucuk_fark = fark
                    en_yakin_j = j
            toplam_fark_a_to_b += agirlik_a[i] * en_kucuk_fark
            eslesmeler_a.append({
                "kaynak_hex": renk_a.hex_kodu,
                "hedef_hex": palet_b[en_yakin_j].hex_kodu,
                "delta_e": round(en_kucuk_fark, 2),
                "agirlik": round(agirlik_a[i], 3)
            })

        # 2. B'den A'ya Ağırlıklı En Yakın Mesafe
        toplam_fark_b_to_a = 0.0
        for j, renk_b in enumerate(palet_b):
            en_kucuk_fark = min(self._iki_renk_mesafesi(renk_b.lab, renk_a.lab) for renk_a in palet_a)
            toplam_fark_b_to_a += agirlik_b[j] * en_kucuk_fark

        # Simetrik İki Yönlü Mesafe
        simetrik_mesafe = (toplam_fark_a_to_b + toplam_fark_b_to_a) / 2.0
        return float(simetrik_mesafe), eslesmeler_a

    def benzerlik_skoru_hesapla(
        self,
        palet_a: List[PaletRengi],
        palet_b: List[PaletRengi]
    ) -> Tuple[float, float, List[Dict[str, Any]]]:
        """Mesafeyi %0 ile %100 arası algısal benzerlik skoruna dönüştürür.

        Döndürür:
            Tuple[float, float, List]: (Benzerlik Skoru %, Delta-E Mesafesi, Eşleşmeler)
        """
        mesafe, eslesmeler = self.palet_mesafesi_hesapla(palet_a, palet_b)
        skor = 100.0 * np.exp(-mesafe / self.sigma)
        return round(float(skor), 2), round(float(mesafe), 2), eslesmeler
