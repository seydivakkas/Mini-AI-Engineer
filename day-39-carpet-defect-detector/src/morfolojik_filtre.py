"""
Morfolojik Filtreleme ve Maske Temizleyici (Morphological Defect Filter).
Açma (Opening), Kapama (Closing) ve Gürültü Eleme.
"""

from typing import Tuple
import numpy as np
from scipy.ndimage import (
    binary_opening,
    binary_closing,
    binary_dilation,
    generate_binary_structure
)


class MorfolojikKusurFiltresi:
    """Ham ikili maske üzerindeki mikro gürültüleri temizleyip kusur sınırlarını birleştirir."""

    def __init__(self, acma_iter: int = 1, kapama_iter: int = 2, min_piksel_alani: int = 25):
        self.acma_iter = acma_iter
        self.kapama_iter = kapama_iter
        self.min_alan = min_piksel_alani

    def temizle_ve_birlestir(self, ham_maske: np.ndarray) -> np.ndarray:
        """
        1. Açma (Opening): Küçük saçılmış piksel gürültülerini yok eder.
        2. Kapama (Closing): Kopuk çizgi parçalarını ve delikleri kapatır.
        3. Dilation: Kusur sınırlarını netleştirir.
        """
        yapi = generate_binary_structure(2, 2)  # 8-bağlantılı 3x3 çekirdek

        # 1. Açma (Erozyon -> Genişleme)
        maske_acik = binary_opening(ham_maske, structure=yapi, iterations=self.acma_iter)

        # 2. Kapama (Genişleme -> Erozyon)
        maske_kapali = binary_closing(maske_acik, structure=yapi, iterations=self.kapama_iter)

        temiz_maske = maske_kapali.astype(np.uint8)
        return temiz_maske
