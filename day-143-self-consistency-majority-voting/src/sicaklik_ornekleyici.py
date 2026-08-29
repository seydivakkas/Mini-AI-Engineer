"""
Sıcaklık Örnekleyici Modülü (Day 143 - Faz 8).
Farklı sıcaklık rejimlerinde (T=0.0, 0.3, 0.7, 1.2) çoklu akıl yürütme yolları ve log-olasılık skorları üreten motor.
"""

from typing import List, Dict, Any
import math
import random


class SicaklikOrnekleyici:
    """Farklı sıcaklık rejimlerinde düşünce yolları ve yol olasılıkları (P(trajectory)) üretir."""

    def __init__(self, tohum: int = 42):
        random.seed(tohum)

        # Farklı akıl yürütme stratejileri ve her birinin baz logitleri
        self._stratejiler = [
            {
                "id": "cebirsel_dogru",
                "adi": "Cebirsel Modelleme",
                "adimlar": ["S + T = 1.10", "S = T + 1.00", "2T = 0.10 => T = 0.05"],
                "tahmin": "0.05",
                "baz_logit": 3.8,  # Yüksek olası doğru yol
            },
            {
                "id": "aritmetik_dogru",
                "adi": "Aritmetik Fark Yolu",
                "adimlar": ["Farkı çıkar: 1.10 - 1.00 = 0.10", "İkiye böl: 0.10 / 2 = 0.05"],
                "tahmin": "0.05",
                "baz_logit": 3.2,
            },
            {
                "id": "denetim_dogru",
                "adi": "Varsayım ve Çelişki Testi",
                "adimlar": ["Top=0.10 olsa toplam 1.20 olurdu", "Düzeltme: Top=0.05 olmalı"],
                "tahmin": "0.05",
                "baz_logit": 2.9,
            },
            {
                "id": "sezgisel_yanlis",
                "adi": "Hızlı Sezgisel Hata",
                "adimlar": ["1.10 - 1.00 = 0.10", "Top doğrudan 0.10 olarak tahmin edildi"],
                "tahmin": "0.10",
                "baz_logit": 1.8,  # Düşük ama sezgisel tuzak
            },
            {
                "id": "kaotik_yanlis",
                "adi": "Kaotik Halüsinasyon",
                "adimlar": ["Top ve sopa eşit kabul edildi", "1.10 / 2 = 0.55"],
                "tahmin": "0.55",
                "baz_logit": 0.5,  # Çok düşük olası sapan
            },
        ]

    def ornekle(self, n_ornek: int = 5, sicaklik: float = 0.7) -> List[Dict[str, Any]]:
        """
        Verilen sıcaklıkta Softmax olasılıkları ile N adet yol örnekler ve log-olasılıkları hesaplar.
        """
        # Sıcaklık T=0.0 ise deterministik (Greedy)
        if sicaklik < 1e-4:
            en_iyi = max(self._stratejiler, key=lambda x: x["baz_logit"])
            return [
                {
                    "ornek_no": i + 1,
                    "strateji": en_iyi["adi"],
                    "adimlar": en_iyi["adimlar"],
                    "tahmin": en_iyi["tahmin"],
                    "sicaklik": 0.0,
                    "yol_olasiligi": 1.0,
                    "log_olasilik": 0.0,
                }
                for i in range(n_ornek)
            ]

        # Softmax hesaplama: P(strat_i) = exp(logit_i / T) / sum(exp(logit_j / T))
        olcekli_logitler = [s["baz_logit"] / sicaklik for s in self._stratejiler]
        maks_logit = max(olcekli_logitler)
        exp_degerler = [math.exp(l - maks_logit) for l in olcekli_logitler]
        toplam_exp = sum(exp_degerler)
        olasiliklar = [e / toplam_exp for e in exp_degerler]

        secilen_yollar = []
        for i in range(n_ornek):
            secim = random.choices(self._stratejiler, weights=olasiliklar, k=1)[0]
            idx = self._stratejiler.index(secim)
            p_yol = olasiliklar[idx]
            log_p = math.log(max(1e-9, p_yol))

            secilen_yollar.append({
                "ornek_no": i + 1,
                "strateji": secim["adi"],
                "adimlar": secim["adimlar"],
                "tahmin": secim["tahmin"],
                "sicaklik": round(sicaklik, 2),
                "yol_olasiligi": round(p_yol, 4),
                "log_olasilik": round(log_p, 4),
            })

        return secilen_yollar
