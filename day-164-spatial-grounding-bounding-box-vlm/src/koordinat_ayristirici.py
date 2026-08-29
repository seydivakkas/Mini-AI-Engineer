"""
Koordinat Ayrıştırıcı ve Normalleştirici Modülü (Day 164 - FAZ 9).
Modelin ürettiği metinsel [ymin, xmin, ymax, xmax] koordinatlarını ayrıştırır ve piksel uzayına dönüştürür.
"""

import re
from typing import List, Tuple, Optional


class KoordinatAyristirici:
    """VLM Koordinat Çıktıları için Regex Ayrıştırıcı ve Normalleştirici."""

    @classmethod
    def metinden_koordinat_cikar(cls, metin: str) -> List[List[int]]:
        """
        Örnek Metin: 'Tespit edilen kırmızı araba: [150, 200, 650, 800] ve kedi [700, 100, 950, 400]'
        Çıktı: [[150, 200, 650, 800], [700, 100, 950, 400]] (0-1000 normalize skala)
        """
        desen = r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
        eslesmeler = re.findall(desen, metin)

        kutular = []
        for e in eslesmeler:
            ymin, xmin, ymax, xmax = map(int, e)
            # Sınır kontrolleri
            ymin = max(0, min(1000, ymin))
            xmin = max(0, min(1000, xmin))
            ymax = max(ymin, min(1000, ymax))
            xmax = max(xmin, min(1000, xmax))
            kutular.append([ymin, xmin, ymax, xmax])

        return kutular

    @classmethod
    def piksel_koordinatina_donustur(
        cls,
        norm_kutu: List[int],
        resim_genislik: int = 640,
        resim_yukseklik: int = 480,
    ) -> List[int]:
        """
        0-1000 normalize kutuyu [ymin, xmin, ymax, xmax] mutlak piksel koordinatlarına dönüştürür.
        """
        ymin, xmin, ymax, xmax = norm_kutu
        px_ymin = int((ymin / 1000.0) * resim_yukseklik)
        px_xmin = int((xmin / 1000.0) * resim_genislik)
        px_ymax = int((ymax / 1000.0) * resim_yukseklik)
        px_xmax = int((xmax / 1000.0) * resim_genislik)
        return [px_ymin, px_xmin, px_ymax, px_xmax]
