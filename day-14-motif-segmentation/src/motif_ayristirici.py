"""Motif ve Desen Ayrıştırma Modülü (Otsu, Kontur ve Şekil Analitiği).

Bu modül; geleneksel kilim, çini veya kumaş gibi yüzeylerdeki geometrik ve
organik motifleri Otsu eşikleme, morfolojik temizleme, kontur hiyerarşisi
ve sınırlayıcı kutular (Bounding Box) ile bağımsız nesnelere ayrıştırır.
"""

from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np


@dataclass
class MotifBilgisi:
    """Ayrıştırılmış tek bir motife ait geometrik ve görsel metrikler."""

    motif_id: int
    alan: float
    cevre: float
    dairesellik: float  # [0.0, 1.0] (1.0 = kusursuz daire)
    doluluk_orani: float  # Solidity (Alan / Dışbükey Gövde Alanı)
    en_boy_orani: float  # Genişlik / Yükseklik
    sinirlayici_kutu: Tuple[int, int, int, int]  # (x, y, w, h)
    dondurulmus_kutu: np.ndarray  # (4, 2) minimum alanlı kutu köşeleri
    merkez: Tuple[float, float]  # Ağırlık merkezi (cx, cy)
    kirpilmis_gorsel: np.ndarray  # Yalnızca motifi içeren BGR kırpıntı
    kirpilmis_maske: np.ndarray  # Motifin ikili alfa maskesi


class MotifAyristirici:
    """Görüntülerden desen ve motifleri tespit eden, izole eden motor."""

    @staticmethod
    def otsu_esikleme(gorsel_gri: np.ndarray, ters_cevir: bool = False) -> Tuple[np.ndarray, float]:
        """Otsu algoritması ile sınıflar arası varyansı maksimize eden eşikleme uygular.

        Parametreler:
            gorsel_gri (np.ndarray): Tek kanallı gri seviye görüntü.
            ters_cevir (bool): True ise koyu motifleri beyaz ön plan yapar (THRESH_BINARY_INV).

        Döndürür:
            Tuple[np.ndarray, float]: (İkili Maske, Optimal Eşik Değeri T*)
        """
        if gorsel_gri.ndim != 2:
            raise ValueError("Otsu eşikleme için tek kanallı gri görüntü gereklidir.")

        bayrak = cv2.THRESH_BINARY_INV if ters_cevir else cv2.THRESH_BINARY
        esik_degeri, maske = cv2.threshold(
            gorsel_gri, 0, 255, bayrak + cv2.THRESH_OTSU
        )
        return maske, float(esik_degeri)

    @staticmethod
    def morfolojik_temizleme(ikili_maske: np.ndarray, cekirdek_boyutu: int = 3) -> np.ndarray:
        """Gürültü noktalarını silmek ve iç delikleri kapatmak için Açma + Kapatma uygular."""
        k = max(1, int(cekirdek_boyutu))
        cekirdek = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

        # Açma (Opening): Küçük izole beyaz noktaları temizler
        acma = cv2.morphologyEx(ikili_maske, cv2.MORPH_OPEN, cekirdek)
        # Kapatma (Closing): Motif içindeki küçük siyah delikleri yamalar
        kapatma = cv2.morphologyEx(acma, cv2.MORPH_CLOSE, cekirdek)
        return kapatma

    @classmethod
    def motifleri_ayristir(
        cls,
        gorsel_bgr: np.ndarray,
        min_alan: float = 200.0,
        maks_alan_orani: float = 0.85,
        morfolojik_cekirdek: int = 3
    ) -> Tuple[List[MotifBilgisi], np.ndarray]:
        """Görseldeki motifleri kontur analitiği ile tespit edip özelliklerini çıkarır.

        Döndürür:
            Tuple[List[MotifBilgisi], np.ndarray]: (Motif Listesi, Temizlenmiş İkili Maske)
        """
        if gorsel_bgr.ndim != 3:
            raise ValueError("Kaynak görüntü 3 kanallı BGR olmalıdır.")

        h_img, w_img = gorsel_bgr.shape[:2]
        toplam_piksel_alani = float(h_img * w_img)

        # 1. Griye çevir ve hafif Gauss bulanıklığı ile yüksek frekanslı doku gürültüsünü süz
        gri = cv2.cvtColor(gorsel_bgr, cv2.COLOR_BGR2GRAY)
        bulanik = cv2.GaussianBlur(gri, (5, 5), 0)

        # 2. Otsu Eşikleme (Motifler zemin rengine göre ayrıştırılır)
        maske_ham, _ = cls.otsu_esikleme(bulanik, ters_cevir=False)

        # Zemin çoğunlukla beyaz mı kontrol et; ters ise çevir
        if np.mean(maske_ham) > 127:
            maske_ham = cv2.bitwise_not(maske_ham)

        # 3. Morfolojik temizleme
        maske_temiz = cls.morfolojik_temizleme(maske_ham, cekirdek_boyutu=morfolojik_cekirdek)

        # 4. Dış konturları bul (RETR_EXTERNAL hiyerarşisi)
        konturlar, _ = cv2.findContours(
            maske_temiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motifler: List[MotifBilgisi] = []
        motif_sayaci = 1

        for cnt in konturlar:
            alan = float(cv2.contourArea(cnt))

            # Çok küçük parazitleri ve tüm resmi kaplayan dış sınırları filtrele
            if alan < min_alan or alan > (toplam_piksel_alani * maks_alan_orani):
                continue

            cevre = float(cv2.arcLength(cnt, closed=True))
            if cevre <= 0:
                continue

            # Dairesellik: 4 * pi * Alan / Cevre^2
            dairesellik = float((4.0 * np.pi * alan) / (cevre**2))
            dairesellik = min(1.0, max(0.0, dairesellik))

            # Doluluk Oranı (Solidity): Alan / Convex Hull Alanı
            govde = cv2.convexHull(cnt)
            govde_alani = float(cv2.contourArea(govde))
            doluluk = float(alan / govde_alani) if govde_alani > 0 else 0.0

            # Sınırlayıcı Kutular
            x, y, w, h = cv2.boundingRect(cnt)
            en_boy = float(w) / float(h) if h > 0 else 1.0

            dondurulmus_dikdortgen = cv2.minAreaRect(cnt)
            kutu_noktalari = cv2.boxPoints(dondurulmus_dikdortgen)

            # Ağırlık Merkezi (Moments)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                cx, cy = float(x + w / 2.0), float(y + h / 2.0)

            # Kırpılmış Görsel ve Maske (ROI)
            kirpilmis_gorsel = gorsel_bgr[y:y + h, x:x + w].copy()
            kirpilmis_maske = maske_temiz[y:y + h, x:x + w].copy()

            motifler.append(
                MotifBilgisi(
                    motif_id=motif_sayaci,
                    alan=round(alan, 1),
                    cevre=round(cevre, 1),
                    dairesellik=round(dairesellik, 3),
                    doluluk_orani=round(doluluk, 3),
                    en_boy_orani=round(en_boy, 3),
                    sinirlayici_kutu=(int(x), int(y), int(w), int(h)),
                    dondurulmus_kutu=kutu_noktalari.astype(np.float32),
                    merkez=(round(cx, 1), round(cy, 1)),
                    kirpilmis_gorsel=kirpilmis_gorsel,
                    kirpilmis_maske=kirpilmis_maske
                )
            )
            motif_sayaci += 1

        # Alana göre büyükten küçüğe sırala
        motifler.sort(key=lambda m: m.alan, reverse=True)
        return motifler, maske_temiz
