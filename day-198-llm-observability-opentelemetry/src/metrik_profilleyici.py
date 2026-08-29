"""
LLM Gözlemlenebilirlik ve İstatistiksel Dağılım Profilleyici Modülü (Day 198 - FAZ 10).
TTFT, TPOT ve Kuyruk Bekleme Sürelerinin P50, P90, P99 Yüzdelik Analitiği.
"""

from typing import Dict, Any, List
import numpy as np
from .opentelemetry_motoru import LLMObservabilityCollector


class LLMGozlemlenebilirlikProfilleyici:
    """
    LLM Çıkarım Metrikleri İstatistiksel Profilleyicisi.
    """

    @classmethod
    def toplu_izleme_profille(cls, trace_sayisi: int = 50) -> Dict[str, Any]:
        """50 adet kurumsal çıkarım izini simüle ederek metrik dağılımlarını hesaplar."""
        ttft_list = []
        tpot_list = []
        queue_list = []
        total_list = []
        ornek_izler = []

        for i in range(trace_sayisi):
            p_len = int(np.random.choice([64, 128, 256, 512, 1024]))
            g_tokens = int(np.random.choice([16, 32, 64, 128]))

            iz_sonucu = LLMObservabilityCollector.inferans_izi_kaydet(prompt_len=p_len, gen_tokens=g_tokens)
            m = iz_sonucu["metrikler"]

            ttft_list.append(m["ttft_ms"])
            tpot_list.append(m["tpot_ms"])
            queue_list.append(m["queue_wait_ms"])
            total_list.append(m["total_latency_ms"])

            if i < 5:
                ornek_izler.append(iz_sonucu)

        def yuzdelik(dizi):
            arr = np.array(dizi)
            return {
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p99": float(np.percentile(arr, 99)),
                "ort": float(np.mean(arr)),
            }

        return {
            "toplam_trace": trace_sayisi,
            "ttft_istatistik": yuzdelik(ttft_list),
            "tpot_istatistik": yuzdelik(tpot_list),
            "queue_istatistik": yuzdelik(queue_list),
            "total_latency_istatistik": yuzdelik(total_list),
            "ornek_iz": ornek_izler[0],
            "ham_veriler": {
                "ttft": ttft_list,
                "tpot": tpot_list,
                "queue": queue_list,
            }
        }
