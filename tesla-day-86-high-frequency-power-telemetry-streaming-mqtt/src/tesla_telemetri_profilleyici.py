"""
Tesla Telemetri Profilleyici Modülü
====================================
Bu modül; 100 Hz telemetri paketleme, serileştirme ve kayan pencere
istatistik çözümleme hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_telemetri_yayinci import TeslaPowerTelemetryStreamer


class TeslaTelemetriProfilleyici:
    """
    Tesla Güç Telemetrisi Performans Profilleyicisi.
    """
    def __init__(self, sample_count: int = 1000):
        self.sample_count = sample_count

    def benchmark_telemetry_stream(self) -> Dict[str, Any]:
        streamer = TeslaPowerTelemetryStreamer(buffer_capacity=1000)

        gecikmeler_us: List[float] = []
        p_samples: List[float] = []

        now_ns = int(time.time_ns())

        for i in range(self.sample_count):
            t_sample = now_ns + i * 10_000_000  # 10 ms aralıklarla (100 Hz)
            v = 400.0 + float(np.sin(i / 10.0) * 5.0)
            cur = 250.0 + float(np.cos(i / 10.0) * 10.0)
            p = (v * cur) / 1000.0
            q = 15.0
            freq = 50.0 + float(np.sin(i / 50.0) * 0.05)
            temp = 45.0 + float(i / 200.0)

            t0 = time.perf_counter_ns()
            raw = streamer.push_sample(t_sample, v, cur, p, q, freq, temp)
            _ = streamer.unpack_telemetry(raw)
            _ = streamer.get_sliding_window_stats()
            t1 = time.perf_counter_ns()

            gecikmeler_us.append(float(t1 - t0) / 1000.0)
            p_samples.append(p)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        stats = streamer.get_sliding_window_stats()

        return {
            "sample_count": self.sample_count,
            "packet_size_bytes": streamer.PACKET_SIZE_BYTES,
            "bandwidth_kb_s": float((streamer.PACKET_SIZE_BYTES * 100) / 1024.0),
            "step_ortalama_us": t_avg_us,
            "step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_isleme_kapasitesi": int(1e6 / max(t_avg_us, 1e-4)),
            "window_mean_kw": stats["mean_kw"],
            "window_max_kw": stats["max_kw"],
            "window_min_kw": stats["min_kw"],
            "p_samples": p_samples[:300],
            "gecikmeler": gecikmeler_us[:200]
        }
