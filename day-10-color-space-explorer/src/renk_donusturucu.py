"""Renk Uzayları Dönüşümü ve Renk Tabanlı Segmentasyon Modülü.

Bu modül; BGR, RGB, HSV, CIELAB ve YCrCb renk uzayları arasındaki dönüşümleri,
bireysel kanal ayrıştırmalarını, OpenCV Ton (Hue) döngüsünü çözen kırmızı segmentasyonunu
ve algısal CIELAB Delta-E renk mesafesi eşiklemesini yürütür.
"""

from typing import Tuple
import cv2
import numpy as np


class RenkUzayiDonusturucu:
    """Renk uzayları dönüşüm ve kanal ayrıştırma sınıfı."""

    @staticmethod
    def bgr_to_rgb(gorsel_bgr: np.ndarray) -> np.ndarray:
        """BGR görüntüyü RGB uzayına çevirir."""
        return cv2.cvtColor(gorsel_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def bgr_to_hsv(gorsel_bgr: np.ndarray) -> np.ndarray:
        """BGR görüntüyü HSV (Ton, Doygunluk, Değer) uzayına çevirir."""
        return cv2.cvtColor(gorsel_bgr, cv2.COLOR_BGR2HSV)

    @staticmethod
    def bgr_to_lab(gorsel_bgr: np.ndarray) -> np.ndarray:
        """BGR görüntüyü CIELAB (Aydınlık, a*, b*) uzayına çevirir."""
        return cv2.cvtColor(gorsel_bgr, cv2.COLOR_BGR2LAB)

    @staticmethod
    def bgr_to_ycrcb(gorsel_bgr: np.ndarray) -> np.ndarray:
        """BGR görüntüyü YCrCb (Lüminans, Kırmızı Fark, Mavi Fark) uzayına çevirir."""
        return cv2.cvtColor(gorsel_bgr, cv2.COLOR_BGR2YCrCb)

    @staticmethod
    def kanallari_ayristir(gorsel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """3 kanallı herhangi bir görüntünün kanallarını tekil 2B matrisler olarak döndürür."""
        if gorsel.ndim != 3 or gorsel.shape[2] != 3:
            raise ValueError("Görüntü 3 kanallı bir matris olmalıdır.")
        k1, k2, k3 = cv2.split(gorsel)
        return k1, k2, k3


class RenkSegmentasyoncu:
    """Işık değişimlerine ve gölgelere dayanıklı renk segmentasyon motoru."""

    @staticmethod
    def hsv_esikleme(
        gorsel_bgr: np.ndarray,
        alt_sinir_hsv: np.ndarray,
        ust_sinir_hsv: np.ndarray
    ) -> np.ndarray:
        """HSV uzayında belirtilen sınırlar arasında kalan pikselleri ikili maske olarak çıkarır."""
        hsv = cv2.cvtColor(gorsel_bgr, cv2.COLOR_BGR2HSV)
        maske = cv2.inRange(hsv, alt_sinir_hsv, ust_sinir_hsv)
        return maske

    @classmethod
    def kirmizi_renk_maskesi(
        cls,
        gorsel_bgr: np.ndarray,
        doygunluk_alt: int = 70,
        parlaklik_alt: int = 50
    ) -> np.ndarray:
        """OpenCV'nin 0-179 Ton (Hue) çemberinde iki uca bölünen kırmızıyı birleşik yakalar.

        Aralık 1: [0, 10] (Kırmızı başlangıcı)
        Aralık 2: [170, 179] (Kırmızı bitişi)
        """
        hsv = cv2.cvtColor(gorsel_bgr, cv2.COLOR_BGR2HSV)

        # 1. Aralık (0 - 10)
        alt_1 = np.array([0, doygunluk_alt, parlaklik_alt], dtype=np.uint8)
        ust_1 = np.array([10, 255, 255], dtype=np.uint8)
        maske_1 = cv2.inRange(hsv, alt_1, ust_1)

        # 2. Aralık (170 - 179)
        alt_2 = np.array([170, doygunluk_alt, parlaklik_alt], dtype=np.uint8)
        ust_2 = np.array([179, 255, 255], dtype=np.uint8)
        maske_2 = cv2.inRange(hsv, alt_2, ust_2)

        # İki maskenin mantıksal VEYA ile birleştirilmesi
        return cv2.bitwise_or(maske_1, maske_2)

    @staticmethod
    def cielab_delta_e_maskesi(
        gorsel_bgr: np.ndarray,
        hedef_renk_bgr: Tuple[int, int, int],
        delta_e_esik: float = 35.0,
        aydinlik_haric_tut: bool = False
    ) -> np.ndarray:
        """İnsan gözünün renk algısına en yakın olan CIELAB Delta-E mesafesiyle segmentasyon yapar.

        Parametreler:
            gorsel_bgr (np.ndarray): BGR formatında kaynak görüntü.
            hedef_renk_bgr (Tuple[int, int, int]): Aranacak hedef renk (M, Y, K).
            delta_e_esik (float): Maksimum renk uzaklık eşiği.
            aydinlik_haric_tut (bool): True ise L* aydınlık kanalını yoksayarak yalnızca
                                       a* ve b* kromatik renk düzleminde mesafe ölçer (Gölgeye tam bağışık!).
        """
        hedef_matris = np.uint8([[list(hedef_renk_bgr)]])
        hedef_lab = cv2.cvtColor(hedef_matris, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]

        lab_gorsel = cv2.cvtColor(gorsel_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        if not aydinlik_haric_tut:
            fark = lab_gorsel - hedef_lab
            delta_e = np.sqrt(np.sum(fark**2, axis=2))
        else:
            # Yalnızca a* ve b* kromatik renk koordinatları: sqrt((a1-a2)^2 + (b1-b2)^2)
            fark_ab = lab_gorsel[:, :, 1:] - hedef_lab[1:]
            delta_e = np.sqrt(np.sum(fark_ab**2, axis=2))

        maske = np.where(delta_e <= delta_e_esik, 255, 0).astype(np.uint8)
        return maske

    @staticmethod
    def maskeyi_uygula(gorsel_bgr: np.ndarray, maske: np.ndarray) -> np.ndarray:
        """İkili maskeyi orijinal görüntü üzerine uygulayarak arka planı karartır."""
        return cv2.bitwise_and(gorsel_bgr, gorsel_bgr, mask=maske)
