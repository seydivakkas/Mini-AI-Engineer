"""GrabCut Ön Plan Segmentasyonu ve Arka Plan Kaldırma Motoru.

Bu modül; Gauss Karışım Modelleri (GMM) ve Çizge Kesme (Graph Cut / Min-Cut Max-Flow)
algoritmasını kullanarak karmaşık ve homojen olmayan arka planlardan nesneleri
kesin sınırlarla ayırır, şeffaf PNG ve yeni arka plan kompozitleri üretir.
"""

from typing import Tuple, List, Optional
import cv2
import numpy as np


class GrabCutAyristirici:
    """GrabCut tabanlı ön plan ayırma ve arka plan değiştirme motoru."""

    def __init__(self) -> None:
        self.arka_plan_modeli = np.zeros((1, 65), np.float64)
        self.on_plan_modeli = np.zeros((1, 65), np.float64)

    def dikdortgen_ile_ayristir(
        self,
        gorsel_bgr: np.ndarray,
        dikdortgen: Tuple[int, int, int, int],
        iterasyon_sayisi: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Kullanıcının belirlediği sınırlayıcı kutu ile GrabCut segmentasyonunu başlatır.

        Parametreler:
            gorsel_bgr (np.ndarray): H x W x 3 BGR kaynak görüntü.
            dikdortgen (Tuple[int, int, int, int]): (x, y, w, h) formatında nesne kutusu.
            iterasyon_sayisi (int): GMM ve Graph Cut optimizasyon döngü adedi.

        Döndürür:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                (İzole Ön Plan BGR, İkili Maske [0, 255], Ham 4-Durumlu Maske [0, 1, 2, 3])
        """
        if gorsel_bgr.ndim != 3:
            raise ValueError("Girdi görüntüsü 3 kanallı BGR olmalıdır.")

        h, w = gorsel_bgr.shape[:2]
        x, y, rect_w, rect_h = dikdortgen

        if x < 0 or y < 0 or x + rect_w > w or y + rect_h > h or rect_w <= 0 or rect_h <= 0:
            raise ValueError(f"Geçersiz sınırlayıcı kutu: {dikdortgen} (Görsel boyutu: {w}x{h})")

        # 4 durumlu maske matrisi
        ham_maske = np.zeros((h, w), dtype=np.uint8)

        # Modelleri sıfırla
        self.arka_plan_modeli = np.zeros((1, 65), np.float64)
        self.on_plan_modeli = np.zeros((1, 65), np.float64)

        # 1. Aşama: Dikdörtgen ile başlatma
        cv2.grabCut(
            gorsel_bgr,
            ham_maske,
            (x, y, rect_w, rect_h),
            self.arka_plan_modeli,
            self.on_plan_modeli,
            iterCount=max(1, int(iterasyon_sayisi)),
            mode=cv2.GC_INIT_WITH_RECT
        )

        # Kesin (1) ve Olası (3) ön plan piksellerini 255 yap
        ikili_maske = np.where(
            (ham_maske == cv2.GC_FGD) | (ham_maske == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        on_plan_bgr = cv2.bitwise_and(gorsel_bgr, gorsel_bgr, mask=ikili_maske)
        return on_plan_bgr, ikili_maske, ham_maske

    def maske_ile_iyilestir(
        self,
        gorsel_bgr: np.ndarray,
        mevcut_ham_maske: np.ndarray,
        kesin_on_plan_noktalari: Optional[List[Tuple[int, int]]] = None,
        kesin_arka_plan_noktalari: Optional[List[Tuple[int, int]]] = None,
        firca_yaricapi: int = 3,
        iterasyon_sayisi: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Kullanıcının fırça darbeleriyle (Strokes) ön/arka plan sınırlarını iyileştirir."""
        ham_maske = mevcut_ham_maske.copy()

        # Kesin Ön Plan Darbeleri (GC_FGD = 1)
        if kesin_on_plan_noktalari:
            for pt in kesin_on_plan_noktalari:
                cv2.circle(ham_maske, pt, firca_yaricapi, cv2.GC_FGD, -1)

        # Kesin Arka Plan Darbeleri (GC_BGD = 0)
        if kesin_arka_plan_noktalari:
            for pt in kesin_arka_plan_noktalari:
                cv2.circle(ham_maske, pt, firca_yaricapi, cv2.GC_BGD, -1)

        # 2. Aşama: Maske ile yinelemeli iyileştirme
        cv2.grabCut(
            gorsel_bgr,
            ham_maske,
            None,
            self.arka_plan_modeli,
            self.on_plan_modeli,
            iterCount=max(1, int(iterasyon_sayisi)),
            mode=cv2.GC_INIT_WITH_MASK
        )

        ikili_maske = np.where(
            (ham_maske == cv2.GC_FGD) | (ham_maske == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        on_plan_bgr = cv2.bitwise_and(gorsel_bgr, gorsel_bgr, mask=ikili_maske)
        return on_plan_bgr, ikili_maske, ham_maske

    @staticmethod
    def seffaf_png_olustur(gorsel_bgr: np.ndarray, ikili_maske: np.ndarray) -> np.ndarray:
        """Ön plan nesnesini 4 kanallı (BGRA) şeffaf alfa kanalına yerleştirir."""
        b, g, r = cv2.split(gorsel_bgr)
        bgra = cv2.merge([b, g, r, ikili_maske])
        return bgra

    @staticmethod
    def arka_plan_degistir(
        on_plan_bgr: np.ndarray,
        ikili_maske: np.ndarray,
        yeni_arka_plan_bgr: np.ndarray,
        kenar_yumusatma_yaricap: int = 3
    ) -> np.ndarray:
        """Ön plan nesnesini yeni bir arka plan görseliyle pürüzsüzce birleştirir (Alfa Kompozisyon)."""
        h, w = on_plan_bgr.shape[:2]
        arka_plan = cv2.resize(yeni_arka_plan_bgr, (w, h))

        # Kenarları yumuşatmak (Feathering) için hafif Gauss bulanıklığı
        if kenar_yumusatma_yaricap > 0:
            k = kenar_yumusatma_yaricap * 2 + 1
            alfa = cv2.GaussianBlur(ikili_maske, (k, k), 0).astype(np.float32) / 255.0
        else:
            alfa = ikili_maske.astype(np.float32) / 255.0

        alfa = np.expand_dims(alfa, axis=2)

        # Doğrusal Alfa Karıştırma: Çıktı = Alfa * ÖnPlan + (1 - Alfa) * ArkaPlan
        kompozit = (alfa * on_plan_bgr.astype(np.float32) + (1.0 - alfa) * arka_plan.astype(np.float32))
        return np.clip(kompozit, 0, 255).astype(np.uint8)
