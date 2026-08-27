"""Görüntü Filtreleme ve Kenar Tespiti Modülü (Konvolüsyon, Gauss, Sobel).

Bu modül; 2B konvolüsyon çekirdeklerini, Gauss tabanlı doğrusal yumuşatmayı ve
Sobel gradyan operatörleri ile yönlü kenar tespitini sayısal taşma korumasıyla uygular.
"""

from typing import Tuple
import cv2
import numpy as np


class KonvolusyonFiltresi:
    """Özelleştirilmiş 2B uzamsal konvolüsyon çekirdek uygulayıcısı."""

    @staticmethod
    def ozel_cekirdek_uygula(
        gorsel: np.ndarray,
        cekirdek: np.ndarray,
        derinlik: int = -1
    ) -> np.ndarray:
        """Herhangi bir 2B çekirdeği (kernel) görüntü üzerine kaydırarak uygular.

        Parametreler:
            gorsel (np.ndarray): Gri tonlamalı veya BGR görüntü.
            cekirdek (np.ndarray): Tek boyutlu (ör. 3x3, 5x5) ağırlık matrisi.
            derinlik (int): Hedef görüntü bit derinliği (-1 orijinalle aynı tutar).
        """
        if not isinstance(gorsel, np.ndarray) or not isinstance(cekirdek, np.ndarray):
            raise TypeError("Görsel ve çekirdek NumPy dizisi olmalıdır.")

        if cekirdek.ndim != 2:
            raise ValueError("Konvolüsyon çekirdeği 2 boyutlu bir matris olmalıdır.")

        return cv2.filter2D(src=gorsel, ddepth=derinlik, kernel=cekirdek)


class GaussBulaniklastirici:
    """2B Gauss Çekirdeği ile Gürültü Azaltma ve Yumuşatma Motoru."""

    @staticmethod
    def bulaniklastir(
        gorsel: np.ndarray,
        cekirdek_boyutu: Tuple[int, int] = (5, 5),
        sigma_x: float = 1.5,
        sigma_y: float = 0.0
    ) -> np.ndarray:
        """Görüntüye Gauss yumuşatması uygular.

        Parametreler:
            gorsel (np.ndarray): İşlenecek görüntü.
            cekirdek_boyutu (Tuple[int, int]): Çekirdek genişlik ve yüksekliği (Tek sayı olmalıdır!).
            sigma_x (float): Yatay yöndeki Gauss standart sapması.
            sigma_y (float): Dikey Gauss standart sapması (0 ise sigma_x ile aynı alınır).
        """
        w, h = cekirdek_boyutu
        if w % 2 == 0 or h % 2 == 0 or w <= 0 or h <= 0:
            raise ValueError("Gauss çekirdek boyutları pozitif tek sayılar olmalıdır (ör. 3, 5, 7).")

        return cv2.GaussianBlur(
            src=gorsel,
            ksize=cekirdek_boyutu,
            sigmaX=sigma_x,
            sigmaY=sigma_y
        )


class SobelKenarTespitEdici:
    """Yatay (Gx) ve Dikey (Gy) Gradyanlar ile Kenar ve Büyüklük Hesaplayıcı."""

    @staticmethod
    def gradyan_hesapla(
        gorsel: np.ndarray,
        cekirdek_boyutu: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sobel gradyanlarını hesaplar ve negatif taşmaları engellemek için float64 kullanır.

        Parametreler:
            gorsel (np.ndarray): Gri tonlamalı görüntü.
            cekirdek_boyutu (int): Sobel çekirdek boyutu (1, 3, 5 veya 7).

        Döndürür:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (Gx_uint8, Gy_uint8, G_magnitut_uint8)
        """
        if cekirdek_boyutu not in [1, 3, 5, 7]:
            raise ValueError("Sobel çekirdek boyutu 1, 3, 5 veya 7 olmalıdır.")

        # Görsel renkli ise gri tonlamaya çevir
        gri = gorsel if gorsel.ndim == 2 else cv2.cvtColor(gorsel, cv2.COLOR_BGR2GRAY)

        # 1. Aşama: Sayısal taşmayı önlemek için CV_64F (float64) ile türev alma
        gx_float = cv2.Sobel(gri, cv2.CV_64F, 1, 0, ksize=cekirdek_boyutu)
        gy_float = cv2.Sobel(gri, cv2.CV_64F, 0, 1, ksize=cekirdek_boyutu)

        # 2. Aşama: Gradyan Büyüklüğü G = sqrt(Gx^2 + Gy^2)
        magnitut_float = np.sqrt(gx_float**2 + gy_float**2)

        # 3. Aşama: uint8 (0-255) aralığına güvenli ölçekleme
        gx_uint8 = cv2.convertScaleAbs(gx_float)
        gy_uint8 = cv2.convertScaleAbs(gy_float)
        magnitut_uint8 = cv2.convertScaleAbs(magnitut_float)

        return gx_uint8, gy_uint8, magnitut_uint8
