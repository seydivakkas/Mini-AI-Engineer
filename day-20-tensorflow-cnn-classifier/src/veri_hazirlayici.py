"""Veri Hazırlayıcı ve Sentetik Veri Üretici Modülü.

Görselleri CNN modelleri için hazırlar; piksel değerlerini [0.0, 1.0] aralığına normalize eder,
stratified olarak train/val/test kümelerine böler ve veri çoğaltma (data augmentation) uygular.
"""

from typing import List, Tuple
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


class VeriHazirlayici:
    """CNN eğitimi için veri seti üretim ve ön işleme yöneticisi."""

    def __init__(self, hedef_boyut: Tuple[int, int] = (64, 64), random_state: int = 42) -> None:
        """Veri hazırlayıcıyı ilklendirir."""
        self.hedef_boyut = hedef_boyut
        self.random_state = random_state
        self.sinif_isimleri = ["Vazo", "Kumaş", "Rozet", "Ahşap"]

    def sentetik_veri_seti_uret(
        self, sinif_basina_ornek: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """4 sınıfta dengeli, varyasyonlu görsel veri seti üretir.

        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: (Görseller float32 [N, H, W, 3], Etiketler int64 [N], Sınıf isimleri).
        """
        gorseller = []
        etiketler = []
        h, w = self.hedef_boyut

        np.random.seed(self.random_state)

        for i in range(sinif_basina_ornek):
            # --- Sınıf 0: Vazo (Kırmızı / Elips Gövde) ---
            img0 = np.full((h, w, 3), (220 + (i % 10), 220, 230), dtype=np.uint8)
            bgr0 = (30 + (i % 8) * 3, 50 + (i % 6) * 4, 180 + (i % 10) * 5)
            gen0 = 12 + (i % 5) * 2
            cv2.ellipse(img0, (w // 2, h // 2 + 5), (gen0, 18), 0, 0, 360, bgr0, -1)
            cv2.rectangle(img0, (w // 2 - gen0 // 2, h // 4), (w // 2 + gen0 // 2, h // 2), bgr0, -1)
            gorseller.append(img0)
            etiketler.append(0)

            # --- Sınıf 1: Kumaş (Mavi / Çizgili Doku) ---
            ton1 = (180 + (i % 8) * 5, 80 + (i % 5) * 4, 20 + (i % 4) * 3)
            img1 = np.full((h, w, 3), ton1, dtype=np.uint8)
            adim = 4 + (i % 4)
            for x in range(0, w, adim):
                cv2.line(img1, (x, 0), (x + 20, h), (max(0, ton1[0] - 35), max(0, ton1[1] - 35), max(0, ton1[2] - 10)), 1)
            gorseller.append(img1)
            etiketler.append(1)

            # --- Sınıf 2: Rozet (Altın Sarı Dairesel / Geometrik) ---
            img2 = np.full((h, w, 3), (30, 30, 35), dtype=np.uint8)
            boyut = 15 + (i % 5) * 2
            cv2.circle(img2, (w // 2, h // 2), boyut, (30, 190 + (i % 8) * 6, 240), -1)
            cv2.circle(img2, (w // 2, h // 2), int(boyut * 0.6), (20, 150, 210), -1)
            cv2.circle(img2, (w // 2, h // 2), int(boyut * 0.3), (255, 255, 255), -1)
            gorseller.append(img2)
            etiketler.append(2)

            # --- Sınıf 3: Ahşap (Kahverengi / Yatay Lifler) ---
            ton3 = (30 + (i % 6) * 4, 60 + (i % 8) * 4, 110 + (i % 10) * 5)
            img3 = np.full((h, w, 3), ton3, dtype=np.uint8)
            for y_line in range(0, h, 3 + (i % 3)):
                cizgi_renk = (max(0, ton3[0] - 15), max(0, ton3[1] - 15), max(0, ton3[2] - 15))
                cv2.line(img3, (0, y_line), (w, y_line), cizgi_renk, 1)
            gorseller.append(img3)
            etiketler.append(3)

        # Görselleri RGB ve [0.0, 1.0] aralığına normalize et
        X = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in gorseller], dtype=np.float32) / 255.0
        y = np.array(etiketler, dtype=np.int64)

        return X, y, self.sinif_isimleri

    def veri_bol(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_orani: float = 0.15,
        test_orani: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Veri setini Stratified (veya küçük veri için standart) olarak Train/Val/Test kümelerine ayırır."""
        test_toplam = val_orani + test_orani
        
        # Sınıf başına örnek sayısını kontrol et
        _, counts = np.unique(y, return_counts=True)
        kullan_stratify = bool(np.min(counts) >= 4)

        X_train, X_gecici, y_train, y_gecici = train_test_split(
            X, y,
            test_size=test_toplam,
            stratify=y if kullan_stratify else None,
            random_state=self.random_state
        )

        val_orani_gecici = val_orani / test_toplam
        _, gecici_counts = np.unique(y_gecici, return_counts=True)
        kullan_stratify_gecici = bool(np.min(gecici_counts) >= 2)

        X_val, X_test, y_val, y_test = train_test_split(
            X_gecici, y_gecici,
            test_size=1.0 - val_orani_gecici,
            stratify=y_gecici if kullan_stratify_gecici else None,
            random_state=self.random_state
        )

        return X_train, y_train, X_val, y_val, X_test, y_test
