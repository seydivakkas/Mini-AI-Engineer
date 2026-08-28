"""
Day 93: Kapsamlı Model Performansı ve Kalibrasyon Metrikleri Hesaplayıcısı
-------------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class KalibrasyonProfili:
    kutu_guvenleri: List[float]
    kutu_dogruluklari: List[float]
    kutu_ornek_sayilari: List[int]
    ece_skoru: float
    brier_skoru: float


@dataclass
class ModelMetrikleri:
    toplam_ornek: int
    dogruluk: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    karisiklik_matrisi: np.ndarray
    sinif_f1_skorlari: Dict[int, float]
    kalibrasyon: KalibrasyonProfili


class MetrikHesaplayici:
    """
    Sınıflandırma modelleri için Accuracy, Macro/Weighted F1, Precision,
    Recall, Confusion Matrix, ECE (Expected Calibration Error) ve Brier Skoru hesaplar.
    """

    def __init__(self, sinif_sayisi: int = 10, ece_kutu_sayisi: int = 10):
        self.sinif_sayisi = sinif_sayisi
        self.ece_kutu_sayisi = ece_kutu_sayisi

    def hesapla(
        self,
        y_gercek: np.ndarray,
        y_tahmin: np.ndarray,
        olasiliklar: Optional[np.ndarray] = None,
    ) -> ModelMetrikleri:
        y_gercek = np.array(y_gercek, dtype=int)
        y_tahmin = np.array(y_tahmin, dtype=int)
        n = len(y_gercek)

        # 1. Genel Doğruluk (Accuracy)
        dogruluk = float(np.mean(y_gercek == y_tahmin)) if n > 0 else 0.0

        # 2. Karışıklık Matrisi (Confusion Matrix)
        karisiklik_matrisi = np.zeros((self.sinif_sayisi, self.sinif_sayisi), dtype=int)
        for g, t in zip(y_gercek, y_tahmin):
            if 0 <= g < self.sinif_sayisi and 0 <= t < self.sinif_sayisi:
                karisiklik_matrisi[g, t] += 1

        # 3. Sınıf Bazlı Precision, Recall, F1
        precision_list, recall_list, f1_list, sinif_f1_dict = [], [], [], {}
        sinif_sayimlari = np.sum(karisiklik_matrisi, axis=1)

        for c in range(self.sinif_sayisi):
            tp = karisiklik_matrisi[c, c]
            fp = np.sum(karisiklik_matrisi[:, c]) - tp
            fn = np.sum(karisiklik_matrisi[c, :]) - tp

            p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

            precision_list.append(p)
            recall_list.append(r)
            f1_list.append(f1)
            sinif_f1_dict[c] = f1

        macro_p = float(np.mean(precision_list))
        macro_r = float(np.mean(recall_list))
        macro_f1 = float(np.mean(f1_list))

        toplam_sayim = max(1, np.sum(sinif_sayimlari))
        weighted_f1 = float(np.sum(np.array(f1_list) * sinif_sayimlari) / toplam_sayim)

        # 4. Kalibrasyon ve ECE Hesabı
        kalibrasyon = self._kalibrasyon_hesapla(y_gercek, y_tahmin, olasiliklar)

        return ModelMetrikleri(
            toplam_ornek=n,
            dogruluk=dogruluk,
            macro_precision=macro_p,
            macro_recall=macro_r,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            karisiklik_matrisi=karisiklik_matrisi,
            sinif_f1_skorlari=sinif_f1_dict,
            kalibrasyon=kalibrasyon,
        )

    def _kalibrasyon_hesapla(
        self,
        y_gercek: np.ndarray,
        y_tahmin: np.ndarray,
        olasiliklar: Optional[np.ndarray],
    ) -> KalibrasyonProfili:
        if olasiliklar is None or len(olasiliklar) == 0:
            return KalibrasyonProfili([], [], [], 0.0, 0.0)

        guvenler = np.max(olasiliklar, axis=-1)
        dogrular = (y_gercek == y_tahmin).astype(float)
        n = len(y_gercek)

        kutu_sinirlari = np.linspace(0, 1, self.ece_kutu_sayisi + 1)
        kutu_guvenleri = []
        kutu_dogruluklari = []
        kutu_sayilari = []
        ece = 0.0

        for i in range(self.ece_kutu_sayisi):
            alt, ust = kutu_sinirlari[i], kutu_sinirlari[i + 1]
            maske = (guvenler >= alt) & (guvenler < ust if i < self.ece_kutu_sayisi - 1 else guvenler <= ust)
            kutu_n = int(np.sum(maske))

            if kutu_n > 0:
                kutu_acc = float(np.mean(dogrular[maske]))
                kutu_conf = float(np.mean(guvenler[maske]))
                ece += (kutu_n / n) * abs(kutu_acc - kutu_conf)
                kutu_dogruluklari.append(kutu_acc)
                kutu_guvenleri.append(kutu_conf)
            else:
                kutu_dogruluklari.append(0.0)
                kutu_guvenleri.append((alt + ust) / 2)
            kutu_sayilari.append(kutu_n)

        # Brier Score (Çok Sınıflı)
        one_hot_y = np.zeros((n, self.sinif_sayisi))
        for idx, y in enumerate(y_gercek):
            if 0 <= y < self.sinif_sayisi:
                one_hot_y[idx, y] = 1.0
        brier = float(np.mean(np.sum((olasiliklar - one_hot_y) ** 2, axis=1)))

        return KalibrasyonProfili(
            kutu_guvenleri=kutu_guvenleri,
            kutu_dogruluklari=kutu_dogruluklari,
            kutu_ornek_sayilari=kutu_sayilari,
            ece_skoru=float(ece),
            brier_skoru=brier,
        )
