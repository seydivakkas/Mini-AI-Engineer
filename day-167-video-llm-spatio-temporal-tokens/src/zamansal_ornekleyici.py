"""
Zamansal Kare Örnekleme Modülü (Temporal Frame Sampler) - (Day 167 - FAZ 9).
Uniform (Düzenli Eşit Aralıklı) ve Adaptive (Harekete Dayalı Dinamik) Kare Örnekleme.
"""

from typing import List, Tuple
import numpy as np


class ZamansalKareOrnekleyici:
    """Video karelerinden optimal zamansal örnekleme yapan modül."""

    @classmethod
    def duzenli_ornekle(cls, toplam_kare: int, ornek_sayisi: int = 8) -> List[int]:
        """Eşit aralıklı düzenli (Uniform) kare indeksleri seçer."""
        if toplam_kare <= ornek_sayisi:
            return list(range(toplam_kare))
        indeksler = np.linspace(0, toplam_kare - 1, ornek_sayisi, dtype=int)
        return indeksler.tolist()

    @classmethod
    def uyarlamali_ornekle(cls, kare_fark_skorlari: List[float], ornek_sayisi: int = 8) -> List[int]:
        """
        Kareler arasındaki optik akış / fark skorlarına göre hareketin yoğun olduğu
        kareleri önceliklendirerek dinamik örnekleme yapar.
        """
        toplam_kare = len(kare_fark_skorlari)
        if toplam_kare <= ornek_sayisi:
            return list(range(toplam_kare))

        # En yüksek hareket farkına sahip kareleri seç
        skorlar_ile_indeks = sorted(enumerate(kare_fark_skorlari), key=lambda x: x[1], reverse=True)
        secilenler = sorted([idx for idx, _ in skorlar_ile_indeks[:ornek_sayisi]])
        return secilenler
