"""
Test-Time Compute Scaling Yasaları Matematiksel Modülü (Day 147 - Faz 8).
Çıkarım zamanı hesaplama bütçesi (N), Power-Law hata fonksiyonu ve doğruluk scaling yasaları.
"""

from typing import Dict, Any, List
import math
import numpy as np


class TestTimeScalingModeli:
    """Test-time compute scaling yasalarını ve güç yasası (Power-Law) eğrilerini modelleyen sınıf."""

    __test__ = False  # PyTest'in test sınıfı olarak toplamasını engeller

    def __init__(self, alfa: float = 0.65, beta: float = 0.42, gama: float = 0.05):
        self.alfa = alfa  # Başlangıç hata katsayısı
        self.beta = beta  # Scaling üssü (Test-time compute verimliliği)
        self.gama = gama  # İndirgenemez asimptotik taban hata

    def hata_hesapla(self, butce_n: float) -> float:
        """
        Güç Yasası Hata Formülü: Hata(N) = alpha * N^(-beta) + gamma
        """
        n = max(1.0, float(butce_n))
        hata = self.alfa * (n ** (-self.beta)) + self.gama
        return float(np.clip(hata, 0.0, 1.0))

    def dogruluk_hesapla(self, butce_n: float) -> float:
        """Doğruluk(N) = 1.0 - Hata(N)"""
        return float(np.clip(1.0 - self.hata_hesapla(butce_n), 0.0, 1.0))

    def butce_taramasi(self, butceler: List[int]) -> List[Dict[str, Any]]:
        """Verilen bütçe listesi için hata ve doğruluk projeksiyonlarını üretir."""
        sonuclar = []
        for b in butceler:
            h = self.hata_hesapla(b)
            acc = self.dogruluk_hesapla(b)
            sonuclar.append({
                "butce_n": b,
                "hata_orani": round(h, 4),
                "dogruluk_orani": round(acc, 4),
                "hesaplama_kat_artisi": f"{b}x",
            })
        return sonuclar
