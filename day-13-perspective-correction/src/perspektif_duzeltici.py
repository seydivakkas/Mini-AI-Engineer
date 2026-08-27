"""Perspektif Düzeltme ve Homografi Dönüşüm Motoru.

Bu modül; açılı/yamuk çekilmiş belgeleri, halıları veya nesneleri 4 köşe noktası
üzerinden homografi matrisi hesaplayarak kuşbakışı (Orthographic / Bird's-Eye View)
düzleme taşır ve açı eğikliğini (Deskewing) düzeltir.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


class PerspektifDuzeltici:
    """4 nokta perspektif dönüşümü ve geometrik düzeltme motoru."""

    @staticmethod
    def noktalari_sirala(noktalar: np.ndarray) -> np.ndarray:
        """Rastgele sıradaki 4 köşe noktasını saat yönünde sıralar.

        Sıralama:
            [0] Sol-Üst (Top-Left)
            [1] Sağ-Üst (Top-Right)
            [2] Sağ-Alt (Bottom-Right)
            [3] Sol-Alt (Bottom-Left)

        Parametreler:
            noktalar (np.ndarray): (4, 2) boyutlu köşe koordinatları.

        Döndürür:
            np.ndarray: (4, 2) boyutlu float32 türünde sıralı koordinatlar.
        """
        pts = np.asarray(noktalar, dtype=np.float32)
        if pts.shape != (4, 2):
            raise ValueError(f"Köşe noktaları (4, 2) boyutunda olmalıdır, alınan: {pts.shape}")

        sirali = np.zeros((4, 2), dtype=np.float32)

        # x + y toplamı: Sol-üst minimum, Sağ-alt maksimum
        toplam = pts.sum(axis=1)
        sirali[0] = pts[np.argmin(toplam)]
        sirali[2] = pts[np.argmax(toplam)]

        # y - x farkı: Sağ-üst minimum, Sol-alt maksimum
        fark = pts[:, 1] - pts[:, 0]
        sirali[1] = pts[np.argmin(fark)]
        sirali[3] = pts[np.argmax(fark)]

        return sirali

    @staticmethod
    def hedef_boyutlari_hesapla(sirali_noktalar: np.ndarray) -> Tuple[int, int]:
        """Sıralı köşe noktalarından bozulmamış kuşbakışı genişlik ve yüksekliği hesaplar.

        Formül:
            Genişlik  = max(||Sağ-Alt - Sol-Alt||, ||Sağ-Üst - Sol-Üst||)
            Yükseklik = max(||Sağ-Üst - Sağ-Alt||, ||Sol-Üst - Sol-Alt||)
        """
        sol_ust, sag_ust, sag_alt, sol_alt = sirali_noktalar

        # Genişlik hesabı
        genislik_alt = np.linalg.norm(sag_alt - sol_alt)
        genislik_ust = np.linalg.norm(sag_ust - sol_ust)
        maks_genislik = max(int(np.round(genislik_alt)), int(np.round(genislik_ust)))

        # Yükseklik hesabı
        yukseklik_sag = np.linalg.norm(sag_ust - sag_alt)
        yukseklik_sol = np.linalg.norm(sol_ust - sol_alt)
        maks_yukseklik = max(int(np.round(yukseklik_sag)), int(np.round(yukseklik_sol)))

        return max(1, maks_genislik), max(1, maks_yukseklik)

    @classmethod
    def dort_nokta_donusumu(
        cls,
        gorsel_bgr: np.ndarray,
        noktalar: np.ndarray,
        hedef_genislik: Optional[int] = None,
        hedef_yukseklik: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Görseldeki 4 köşeyi homografi matrisi ile düzleştirir.

        Parametreler:
            gorsel_bgr (np.ndarray): Kaynak görüntü (H x W x C).
            noktalar (np.ndarray): Nesnenin 4 köşe koordinatı (4, 2).
            hedef_genislik (int, opsiyonel): İstenen çıktı genişliği.
            hedef_yukseklik (int, opsiyonel): İstenen çıktı yüksekliği.

        Döndürür:
            Tuple[np.ndarray, np.ndarray]: (Düzeltilmiş Görüntü, 3x3 Homografi Matrisi)
        """
        sirali = cls.noktalari_sirala(noktalar)

        if hedef_genislik is None or hedef_yukseklik is None:
            w, h = cls.hedef_boyutlari_hesapla(sirali)
            hedef_w = hedef_genislik or w
            hedef_h = hedef_yukseklik or h
        else:
            hedef_w, hedef_h = int(hedef_genislik), int(hedef_yukseklik)

        # Hedef düzlem koordinatları (Kuşbakışı dikdörtgen)
        hedef_noktalar = np.array([
            [0, 0],
            [hedef_w - 1, 0],
            [hedef_w - 1, hedef_h - 1],
            [0, hedef_h - 1]
        ], dtype=np.float32)

        # 3x3 Homografi Projeksiyon Matrisinin Hesaplanması
        homografi_matrisi = cv2.getPerspectiveTransform(sirali, hedef_noktalar)

        # Tersine eşleme ve bi-kübik interpolasyon ile piksel projeksiyonu
        duzeltilmis_gorsel = cv2.warpPerspective(
            gorsel_bgr,
            homografi_matrisi,
            (hedef_w, hedef_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return duzeltilmis_gorsel, homografi_matrisi

    @staticmethod
    def egim_acisi_duzelt(gorsel_bgr: np.ndarray, aci_derece: float) -> np.ndarray:
        """Görseli merkez ekseni etrafında belirtilen açı kadar döndürerek eğikliği giderir."""
        h, w = gorsel_bgr.shape[:2]
        merkez = (w / 2.0, h / 2.0)

        # 2x3 Affin Dönüşüm Matrisi
        rotasyon_matrisi = cv2.getRotationMatrix2D(merkez, aci_derece, scale=1.0)

        # Sınırların kesilmemesi için yeni boyut hesabı
        cos_val = np.abs(rotasyon_matrisi[0, 0])
        sin_val = np.abs(rotasyon_matrisi[0, 1])
        yeni_w = int((h * sin_val) + (w * cos_val))
        yeni_h = int((h * cos_val) + (w * sin_val))

        rotasyon_matrisi[0, 2] += (yeni_w / 2.0) - merkez[0]
        rotasyon_matrisi[1, 2] += (yeni_h / 2.0) - merkez[1]

        return cv2.warpAffine(
            gorsel_bgr,
            rotasyon_matrisi,
            (yeni_w, yeni_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
