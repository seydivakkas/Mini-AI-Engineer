"""
Canary Kademeli Geçiş ve Shadow Analiz Profilleyici Modülü (Day 199 - FAZ 10).
A/B Testi, Gölge Trafik Karşılaştırması ve Otomatik Geri Alma (Rollback) Simülasyonu.
"""

from typing import Dict, Any, List
import numpy as np
from .canary_shadow_motoru import (
    LLMModelInstance,
    ShadowTrafficMirror,
    CanaryTrafficRouter,
    CanaryCircuitBreaker,
)


class CanaryGecisProfilleyici:
    """
    Canary Kademeli Geçiş ve Rollback Profilleyicisi.
    """

    @classmethod
    def kademeli_canary_gecis_simulasyonu(cls) -> Dict[str, Any]:
        """4 Aşamalı (%5 -> %20 -> %50 -> %100) başarılı Canary geçiş simülasyonu."""
        baseline = LLMModelInstance("llama3-70b-v1", "v1.0.0", is_canary=False, base_latency_ms=28.0)
        canary = LLMModelInstance("llama3-70b-v2", "v2.0.0", is_canary=True, base_latency_ms=22.0)  # %21 daha hızlı

        router = CanaryTrafficRouter(baseline, canary, canary_weight=0.05)
        asamalar = [
            {"asama": "1. Aşama (%5 Canary)", "agirlik": 0.05, "istek_sayisi": 100},
            {"asama": "2. Aşama (%20 Canary)", "agirlik": 0.20, "istek_sayisi": 100},
            {"asama": "3. Aşama (%50 Canary)", "agirlik": 0.50, "istek_sayisi": 100},
            {"asama": "4. Aşama (%100 Tam Geçiş)", "agirlik": 1.00, "istek_sayisi": 100},
        ]

        asama_raporlari = []
        for a in asamalar:
            router.set_weight(a["agirlik"])
            canary_count = 0
            baseline_count = 0

            for i in range(a["istek_sayisi"]):
                _, is_canary = router.route_request(f"Test prompt {i}")
                if is_canary:
                    canary_count += 1
                else:
                    baseline_count += 1

            asama_raporlari.append({
                "asama_adi": a["asama"],
                "hedef_canary_yuzde": f"%{a['agirlik']*100:.0f}",
                "canary_istek_sayisi": canary_count,
                "baseline_istek_sayisi": baseline_count,
                "gerceklesen_canary_yuzde": (canary_count / a["istek_sayisi"]) * 100.0,
            })

        return {
            "durum": "BAŞARILI GEÇİŞ",
            "asama_raporlari": asama_raporlari,
        }

    @classmethod
    def anomali_ve_otomatik_rollback_simulasyonu(cls) -> Dict[str, Any]:
        """Canary modelinde hata patlaması simüle ederek otomatik rollback'i doğrular."""
        baseline = LLMModelInstance("llama3-70b-v1", "v1.0.0", is_canary=False, base_latency_ms=28.0)
        # Hatalı canary adayı (%10 hata olasılığı)
        bozuk_canary = LLMModelInstance("llama3-70b-v2-buggy", "v2.0.1-rc", is_canary=True, error_prob=0.10)

        router = CanaryTrafficRouter(baseline, bozuk_canary, canary_weight=0.20)
        breaker = CanaryCircuitBreaker(router, max_error_rate=0.03)

        logs = []
        rollback_tetiklendi_mi = False

        for i in range(50):
            resp, is_canary = router.route_request(f"Critical query {i}")
            tripped = breaker.check_and_enforce()
            if tripped and not rollback_tetiklendi_mi:
                rollback_tetiklendi_mi = True

            logs.append({"adim": i, "is_canary": is_canary, "success": resp.get("success", False)})

        return {
            "rollback_tetiklendi": rollback_tetiklendi_mi,
            "canary_son_agirlik": router.canary_weight,
            "toplam_canary_istek": bozuk_canary.total_requests,
            "hatali_canary_istek": bozuk_canary.failed_requests,
        }
