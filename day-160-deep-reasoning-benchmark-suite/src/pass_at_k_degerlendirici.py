"""
Pass@k ve Çoğunluk Oylaması (Self-Consistency) Metrik Motoru (Day 160 - FAZ 8 BÜYÜK FİNALİ).
Unbiased Pass@k formülü ve test-time compute ölçekleme analizini yürütür.
"""

import math
from typing import Dict, Any, List
import numpy as np


class PassAtKDegerlendirici:
    """Pass@k tarafsız tahmincisi ve çoğunluk oylama motoru."""

    @classmethod
    def pass_at_k_hesapla(cls, n: int, c: int, k: int) -> float:
        """
        Unbiased Pass@k formülü: 1 - comb(n - c, k) / comb(n, k)
        n: Toplam örneklem sayısı
        c: Doğru çözüm sayısı
        k: Seçilen deneme bütçesi (k <= n)
        """
        if n - c < k:
            return 1.0
        if c == 0:
            return 0.0

        comb_total = math.comb(n, k)
        comb_incorrect = math.comb(n - c, k)
        return float(1.0 - (comb_incorrect / comb_total))

    @classmethod
    def orneklem_degerlendir(cls, ornek_sonuclari: List[bool], k_degerleri: List[int] = [1, 4, 16]) -> Dict[str, float]:
        """
        Boolean doğruluk listesi üzerinden farklı k değerleri için Pass@k döner.
        """
        n = len(ornek_sonuclari)
        c = sum(1 for s in ornek_sonuclari if s)

        skorlar = {}
        for k in k_degerleri:
            if k <= n:
                skorlar[f"pass@{k}"] = round(cls.pass_at_k_hesapla(n, c, k) * 100.0, 1)

        # Çoğunluk Oylaması (Majority Voting / Self-Consistency)
        skorlar["majority_vote_acc"] = round(100.0 if c > n / 2 else 0.0, 1)
        skorlar["dogru_ornek_sayisi"] = c
        skorlar["toplam_ornek_sayisi"] = n

        return skorlar
