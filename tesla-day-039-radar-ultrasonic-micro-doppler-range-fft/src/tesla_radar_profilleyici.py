"""
Tesla Radar ve Ultrasonik Profilleyici Modülü
==============================================
Bu modül; 2D Range-Doppler FFT hesaplama hızını, CA-CFAR eşikleme
hassasiyetini ve Ultrasonik ToF mesafe gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_radar_ve_ultrasonik_isleme import TeslaRadarAndUltrasonicProcessor


class TeslaRadarProfilleyici:
    """
    Radar ve Ultrasonik Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_radar_ve_ultrasonik(self) -> Dict[str, Any]:
        processor = TeslaRadarAndUltrasonicProcessor()

        # 25 metre mesafede, -15 m/s yaklaşan araç için sentetik radar karesi
        raw_radar = processor.generate_synthetic_radar_frame(target_range_m=25.0, target_speed_mps=-15.0, snr_db=25.0)

        gecikmeler_us: List[float] = []

        rd_map = None
        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            rd_map = processor.compute_range_doppler_fft(raw_radar)
            # 1D Range profili üzerinden CA-CFAR
            range_profile = np.max(rd_map, axis=0)
            detections = processor.ca_cfar_1d(range_profile)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        # Ultrasonik ToF: 1.5 metre için yankı süresi (yaklaşık 8.7 ms)
        t_echo = (2.0 * 1.5) / 343.0
        us_dist_20c = processor.compute_ultrasonic_distance(t_echo, ambient_temp_c=20.0)
        us_dist_minus10c = processor.compute_ultrasonic_distance(t_echo, ambient_temp_c=-10.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "radar_step_ortalama_us": t_avg_us,
            "radar_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_radar_karesi": int(1e6 / max(t_avg_us, 1e-4)),
            "rd_map": rd_map,
            "range_profile": np.max(rd_map, axis=0),
            "doppler_profile": np.max(rd_map, axis=1),
            "us_dist_20c": us_dist_20c,
            "us_dist_minus10c": us_dist_minus10c,
            "gecikmeler": gecikmeler_us[:200]
        }
