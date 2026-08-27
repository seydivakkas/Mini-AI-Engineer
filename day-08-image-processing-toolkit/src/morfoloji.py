"""Matematiksel Morfoloji Modülü (Aşınma, Genişleme, Açma, Kapatma).

Bu modül; ikili (binary) ve gri seviye görüntüler üzerinde nesne sınırlarını inceltme,
genişletme, arka plan gürültülerini süpürme (Açma) ve nesne içi delikleri doldurma (Kapatma)
operasyonlarını yapısal elementler (Structuring Element) üzerinden yürütür.
"""

from typing import Tuple
import cv2
import numpy as np


class MorfolojikIslemci:
    """Matematiksel morfoloji operasyonları yürütücüsü."""

    @staticmethod
    def yapisal_element_olustur(
        boyut: Tuple[int, int] = (3, 3),
        sekil: str = "dikdortgen"
    ) -> np.ndarray:
        """Belirtilen geometride yapısal element (kernel) matrisi üretir.

        Parametreler:
            boyut (Tuple[int, int]): Çekirdek genişliği ve yüksekliği.
            sekil (str): 'dikdortgen', 'elips' veya 'arti'.
        """
        sekil_haritasi = {
            "dikdortgen": cv2.MORPH_RECT,
            "elips": cv2.MORPH_ELLIPSE,
            "arti": cv2.MORPH_CROSS,
        }
        if sekil not in sekil_haritasi:
            raise ValueError(f"Desteklenmeyen şekil: {sekil}. Seçenekler: {list(sekil_haritasi.keys())}")

        return cv2.getStructuringElement(sekil_haritasi[sekil], boyut)

    @classmethod
    def asinma(
        cls,
        gorsel: np.ndarray,
        cekirdek: np.ndarray = None,
        yineleme: int = 1
    ) -> np.ndarray:
        """Aşınma (Erosion): Ön plan piksellerini inceltir, minik beyaz gürültüleri yok eder."""
        if cekirdek is None:
            cekirdek = cls.yapisal_element_olustur((3, 3), "dikdortgen")
        return cv2.erode(gorsel, cekirdek, iterations=yineleme)

    @classmethod
    def genisleme(
        cls,
        gorsel: np.ndarray,
        cekirdek: np.ndarray = None,
        yineleme: int = 1
    ) -> np.ndarray:
        """Genişleme (Dilation): Ön plan sınırlarını büyütür, kopuk çizgileri birleştirir."""
        if cekirdek is None:
            cekirdek = cls.yapisal_element_olustur((3, 3), "dikdortgen")
        return cv2.dilate(gorsel, cekirdek, iterations=yineleme)

    @classmethod
    def acma(
        cls,
        gorsel: np.ndarray,
        cekirdek: np.ndarray = None
    ) -> np.ndarray:
        """Açma (Opening): Önce Aşınma, sonra Genişleme. Arka plan gürültülerini temizler."""
        if cekirdek is None:
            cekirdek = cls.yapisal_element_olustur((3, 3), "dikdortgen")
        return cv2.morphologyEx(gorsel, cv2.MORPH_OPEN, cekirdek)

    @classmethod
    def kapatma(
        cls,
        gorsel: np.ndarray,
        cekirdek: np.ndarray = None
    ) -> np.ndarray:
        """Kapatma (Closing): Önce Genişleme, sonra Aşınma. Nesne içi siyah delikleri tıkar."""
        if cekirdek is None:
            cekirdek = cls.yapisal_element_olustur((3, 3), "dikdortgen")
        return cv2.morphologyEx(gorsel, cv2.MORPH_CLOSE, cekirdek)

    @classmethod
    def morfolojik_gradyan(
        cls,
        gorsel: np.ndarray,
        cekirdek: np.ndarray = None
    ) -> np.ndarray:
        """Morfolojik Gradyan: Genişleme eksi Aşınma. Nesnenin dış kontur çizgisini çıkarır."""
        if cekirdek is None:
            cekirdek = cls.yapisal_element_olustur((3, 3), "dikdortgen")
        return cv2.morphologyEx(gorsel, cv2.MORPH_GRADIENT, cekirdek)
