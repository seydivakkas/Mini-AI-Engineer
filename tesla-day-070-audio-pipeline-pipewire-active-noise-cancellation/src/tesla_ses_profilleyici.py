"""
Tesla Ses Profilleyici Modülü
=============================
Bu modül; ARNC ters faz ses işleme süresini, PipeWire tampon gecikmesini
ve dB gürültü sönümleme verimini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_ses_motoru_arnc import TeslaARNCNoiseCanceller, TeslaMultiZoneAudioRouter


class TeslaSesProfilleyici:
    """
    Tesla ARNC Ses Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_audio_dsp(self) -> Dict[str, Any]:
        canceller = TeslaARNCNoiseCanceller()
        router = TeslaMultiZoneAudioRouter()

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = canceller.process_noise_reduction(frames=480)
            _ = router.route_audio_stream("autopilot_chime")
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "dsp_step_ortalama_us": t_avg_us,
            "dsp_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_dsp_tamponu": int(1e6 / max(t_avg_us, 1e-4)),
            "db_reduction": ciktilar["db_reduction"],
            "latency_ms": ciktilar["buffer_latency_ms"],
            "raw_noise": ciktilar["raw_noise"],
            "anti_noise": ciktilar["anti_noise"],
            "residual": ciktilar["residual_noise"],
            "is_effective": ciktilar["is_effective"],
            "gecikmeler": gecikmeler_us[:200]
        }
