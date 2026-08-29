"""
1F1B ve GPipe Zaman Çizelgesi (Schedule) Simülasyon Motoru (Day 185 - FAZ 10).
GPipe vs 1F1B vs Interleaved 1F1B Balon Oranı ve Aktivasyon Bellek Analitiği.
"""

from typing import List, Dict, Any, Tuple, Optional
from enum import Enum


class ZamanCizelgesiTuru(Enum):
    """Pipeline Zaman Çizelgesi Türleri."""
    NAIVE_GPIPE = "NAIVE_GPIPE"             # Klasik GPipe (Tüm İleriler -> Tüm Geriler)
    SCHEDULE_1F1B = "SCHEDULE_1F1B"         # 1F1B (Warmup -> 1 İleri 1 Geri -> Cooldown)
    INTERLEAVED_1F1B = "INTERLEAVED_1F1B"   # Sanal Aşamalı Interleaved 1F1B (v=2)


class PipelineZamanCizelgesiMotoru:
    """
    Pipeline Parallelism Zaman Çizelgesi ve Balon Simülatörü.
    P aşama (stages) ve M mikro-batch (micro-batches) üzerinde Gantt çizelgesini modeller.
    """

    @classmethod
    def balon_orani_hesapla(
        cls,
        num_stages: int = 8,
        num_microbatches: int = 32,
        virtual_stages: int = 1,
    ) -> float:
        """
        Pipeline Balon (Bubble) Kesir Oranı:
        - Standart GPipe & 1F1B: (P - 1) / (M + P - 1)
        - Interleaved 1F1B (v > 1): (P - 1) / (v * M)
        """
        P = float(num_stages)
        M = float(num_microbatches)
        v = float(virtual_stages)

        if v > 1:
            bubble = (P - 1.0) / (v * M)
        else:
            bubble = (P - 1.0) / (M + P - 1.0)

        return min(max(bubble, 0.0), 1.0)

    @classmethod
    def tepe_aktivasyon_bellegi_mb(
        cls,
        cizelge_turu: ZamanCizelgesiTuru,
        num_stages: int = 8,
        num_microbatches: int = 32,
        microbatch_aktivasyon_mb: float = 250.0,
    ) -> float:
        """
        Aşama 0 üzerindeki tepe aktivasyon bellek tüketimi (MB):
        - GPipe: O(M) -> M * mikrobatch_aktivasyon_mb
        - 1F1B: O(P) -> P * mikrobatch_aktivasyon_mb
        - Interleaved 1F1B: O(P) -> P * mikrobatch_aktivasyon_mb
        """
        if cizelge_turu == ZamanCizelgesiTuru.NAIVE_GPIPE:
            aktif_mb_sayisi = num_microbatches
        else:
            aktif_mb_sayisi = num_stages

        return aktif_mb_sayisi * microbatch_aktivasyon_mb

    @classmethod
    def karsilastirmali_cizelge_raporu(
        cls,
        num_stages: int = 8,
        num_microbatches: int = 32,
        microbatch_aktivasyon_mb: float = 250.0,
    ) -> List[Dict[str, Any]]:
        """GPipe, 1F1B ve Interleaved 1F1B için karşılaştırmalı analitik rapor."""
        senaryolar = [
            {
                "tur": ZamanCizelgesiTuru.NAIVE_GPIPE,
                "ad": "Klasik GPipe (Huang et al.)",
                "virtual_stages": 1,
                "bellek_karmasikligi": "O(M)",
            },
            {
                "tur": ZamanCizelgesiTuru.SCHEDULE_1F1B,
                "ad": "1F1B Steady-State (PipeDream)",
                "virtual_stages": 1,
                "bellek_karmasikligi": "O(P)",
            },
            {
                "tur": ZamanCizelgesiTuru.INTERLEAVED_1F1B,
                "ad": "Interleaved 1F1B (Megatron v=2)",
                "virtual_stages": 2,
                "bellek_karmasikligi": "O(P)",
            },
        ]

        rapor = []
        for s in senaryolar:
            bubble = cls.balon_orani_hesapla(
                num_stages=num_stages,
                num_microbatches=num_microbatches,
                virtual_stages=s["virtual_stages"],
            )
            vram_mb = cls.tepe_aktivasyon_bellegi_mb(
                cizelge_turu=s["tur"],
                num_stages=num_stages,
                num_microbatches=num_microbatches,
                microbatch_aktivasyon_mb=microbatch_aktivasyon_mb,
            )
            rapor.append({
                "cizelge_adi": s["ad"],
                "balon_orani_yuzde": round(bubble * 100.0, 1),
                "tepe_aktivasyon_gb": round(vram_mb / 1024.0, 2),
                "bellek_karmasikligi": s["bellek_karmasikligi"],
                "virtual_stages": s["virtual_stages"],
            })

        return rapor
