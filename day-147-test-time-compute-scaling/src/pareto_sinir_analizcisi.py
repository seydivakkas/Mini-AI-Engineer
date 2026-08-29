"""
Pareto Sınır Analizcisi Modülü (Day 147 - Faz 8).
Küçük Model + Yüksek Test-Time Compute vs Büyük Model + Düşük Compute Pareto analizi.
"""

from typing import List, Dict, Any
import numpy as np


class ParetoSinirAnalizcisi:
    """Farklı model boyutları ve çıkarım bütçeleri arasındaki Pareto verimlilik sınırını hesaplar."""

    @classmethod
    def pareto_karsilastirmasi(cls) -> List[Dict[str, Any]]:
        """
        8B ve 70B modellerin 1x, 4x, 16x ve 64x test-time compute maliyet ve doğruluk verilerini döner.
        """
        senaryolar = [
            # 8B Modeller
            {"model": "8B", "test_compute": 1, "maliyet_birimi": 1.0, "bellek_gb": 16, "dogruluk": 0.52},
            {"model": "8B", "test_compute": 4, "maliyet_birimi": 4.0, "bellek_gb": 16, "dogruluk": 0.68},
            {"model": "8B", "test_compute": 16, "maliyet_birimi": 16.0, "bellek_gb": 16, "dogruluk": 0.81},
            {"model": "8B", "test_compute": 64, "maliyet_birimi": 64.0, "bellek_gb": 16, "dogruluk": 0.89},

            # 70B Modeller
            {"model": "70B", "test_compute": 1, "maliyet_birimi": 9.0, "bellek_gb": 140, "dogruluk": 0.76},
            {"model": "70B", "test_compute": 4, "maliyet_birimi": 36.0, "bellek_gb": 140, "dogruluk": 0.86},
            {"model": "70B", "test_compute": 16, "maliyet_birimi": 144.0, "bellek_gb": 140, "dogruluk": 0.93},
            {"model": "70B", "test_compute": 64, "maliyet_birimi": 576.0, "bellek_gb": 140, "dogruluk": 0.96},
        ]

        # Pareto optimal kontrolü (Daha düşük maliyetle daha yüksek veya eşit doğruluk)
        for s in senaryolar:
            # Kendisinden hem daha ucuz hem daha doğru başka bir senaryo var mı?
            baskilayan_var_mi = any(
                d["maliyet_birimi"] <= s["maliyet_birimi"] and d["dogruluk"] > s["dogruluk"]
                for d in senaryolar if d != s
            )
            s["pareto_optimal_mi"] = not baskilayan_var_mi

        return senaryolar
