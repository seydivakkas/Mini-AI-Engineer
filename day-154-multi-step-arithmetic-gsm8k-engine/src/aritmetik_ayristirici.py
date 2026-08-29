"""
GSM8K & MATH Aritmetik Problem Ayrıştırıcı Modülü (Day 154 - Faz 8).
Problem metnindeki sayısal verileri, birimleri ve hedef soru kökünü ayıklar.
"""

import re
from typing import Dict, Any, List


class AritmetikAyristirici:
    """Sözel matematik problemlerini yapılandırılmış veri nesnesine dönüştürür."""

    @classmethod
    def ayristir(cls, problem_metni: str) -> Dict[str, Any]:
        """
        Problem metnindeki sayıları ve anahtar ifadeleri çıkarır.
        """
        # Sayıları tespit et (tamsayı ve ondalık)
        sayilar = [float(s) if "." in s else int(s) for s in re.findall(r"\b\d+(?:\.\d+)?\b", problem_metni)]

        # Soru cümlesini tespit et
        cumleler = [c.strip() for c in problem_metni.replace("?", "?\n").split("\n") if c.strip()]
        soru_cumlesi = cumleler[-1] if cumleler else problem_metni

        return {
            "ham_problem": problem_metni,
            "tespit_edilen_sayilar": sayilar,
            "soru_cumlesi": soru_cumlesi,
            "sayi_adedi": len(sayilar),
        }
