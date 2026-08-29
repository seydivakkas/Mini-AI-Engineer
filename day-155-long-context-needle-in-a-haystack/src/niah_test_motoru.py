"""
Needle In A Haystack (NIAH) Test ve Isı Haritası Değerlendirici Motoru (Day 155 - Faz 8).
Bağlam uzunluğu ve derinlik ızgarasında geri çağırma doğruluğunu ve 'Lost in the Middle' etkisini hesaplar.
"""

import numpy as np
from typing import Dict, Any, List, Tuple


class NIAHTestMotoru:
    """NIAH ızgara testlerini koşturan ve doğruluk matrisini üreten değerlendirme motoru."""

    def __init__(
        self,
        baglam_uzunluklari: List[int] = None,
        derinlik_yuzdeleri: List[int] = None,
    ):
        self.baglam_uzunluklari = baglam_uzunluklari or [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
        self.derinlik_yuzdeleri = derinlik_yuzdeleri or [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    def izgara_matrisini_hesapla(self) -> np.ndarray:
        """
        Bağlam uzunlukları (satırlar) ve derinlik yüzdeleri (sütunlar) için
        'Lost in the Middle' ve RoPE zayıflamasını modelleyen doğruluk matrisi üretir.
        """
        n_satir = len(self.baglam_uzunluklari)
        n_sutun = len(self.derinlik_yuzdeleri)
        matris = np.ones((n_satir, n_sutun), dtype=float)

        for i, uzunluk in enumerate(self.baglam_uzunluklari):
            for j, derinlik in enumerate(self.derinlik_yuzdeleri):
                # 1. Küçük bağlamlarda (<16k) doğruluk %100'e yakındır
                if uzunluk <= 16000:
                    matris[i, j] = 1.0
                else:
                    # 2. 'Lost in the Middle' etkisi: Derinlik %40-%60 arasında düşüş yaşanır
                    ortaya_uzaklik = abs(derinlik - 50.0) / 50.0 # 0.0 (tam ortada) -> 1.0 (uçlarda)
                    uzunluk_carpani = (uzunluk / 128000.0)

                    # U-şekilli doğruluk eğrisi: Uçlar (%0 ve %100) yüksek, orta (%50) düşük
                    temel_dogruluk = 0.65 + (0.35 * ortaya_uzaklik)
                    bozulma = 0.25 * uzunluk_carpani * (1.0 - ortaya_uzaklik)

                    dogruluk = max(0.2, min(1.0, temel_dogruluk - bozulma))
                    matris[i, j] = round(dogruluk, 2)

        return matris

    def tam_degerlendirme_yap(self) -> Dict[str, Any]:
        """
        Tüm ızgarayı çalıştırır ve istatistiksel özet üretir.
        """
        matris = self.izgara_matrisini_hesapla()
        ortalama_dogruluk = float(np.mean(matris))
        orta_bolge_dogruluk = float(np.mean(matris[:, 4:7])) # %40-%60 arası

        return {
            "baglam_uzunluklari": self.baglam_uzunluklari,
            "derinlik_yuzdeleri": self.derinlik_yuzdeleri,
            "dogruluk_matrisi": matris,
            "ortalama_dogruluk": ortalama_dogruluk,
            "orta_bolge_dogruluk": orta_bolge_dogruluk,
            "lost_in_middle_kaybi": round(ortalama_dogruluk - orta_bolge_dogruluk, 3),
        }
