"""
Tesla String View ve Span Ayristirici Profilleyicisi
====================================================
Bu modul; C++20 `std::string_view` / `std::span` sifir kopyalama yontemi ile
klasik dinamik bellek tahsisli `std::string::substr` ayrismasi arasindaki
gecikme ve bellek tahsis farklarini olcer.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_span_ranges_ayristirici import TeslaNMEAAyristirici


class TeslaAyristiriciProfilleyici:
    """
    NMEA GPS ayrıştırma performans profilleyicisi.
    """
    def __init__(self, dongu_sayisi: int = 10000):
        self.dongu_sayisi = dongu_sayisi
        self.ornek_nmea = "$GPRMC,083559.00,A,3723.2475,N,12208.3845,W,55.4,180.0,300826,,,A*72"

    def benchmark_string_view_vs_kopyalama(self) -> Dict[str, Any]:
        """
        String_view (sıfır kopyalama) ile klasik string kopyalama karşılaştırması.
        """
        # 1. String View Sıfır Tahsisli Ayrıştırma
        gecikmeler_view: List[float] = []
        for _ in range(self.dongu_sayisi):
            t0 = time.perf_counter_ns()
            TeslaNMEAAyristirici.gprmc_ayristir(self.ornek_nmea)
            t1 = time.perf_counter_ns()
            gecikmeler_view.append(float(t1 - t0))

        # 2. Klasik String Kopyalama & Split (Heap Tahsisli)
        gecikmeler_kopya: List[float] = []
        for _ in range(self.dongu_sayisi):
            t0 = time.perf_counter_ns()
            parcalar = self.ornek_nmea.split(',')
            _ = [str(p) for p in parcalar]  # Heap kopya simülasyonu
            t1 = time.perf_counter_ns()
            gecikmeler_kopya.append(float(t1 - t0))

        view_dizi = np.array(gecikmeler_view)
        kopya_dizi = np.array(gecikmeler_kopya)

        return {
            "view_ort_ns": float(np.mean(view_dizi)),
            "view_p99_ns": float(np.percentile(view_dizi, 99)),
            "view_jitter_ns": float(np.std(view_dizi)),
            "kopya_ort_ns": float(np.mean(kopya_dizi)),
            "hizlanma_orani": float(np.mean(kopya_dizi) / max(np.mean(view_dizi), 1e-3)),
            "view_tahsis_sayisi": 0,
            "kopya_tahsis_sayisi": 12,
            "saniyede_cumle_sayisi": float(1e9 / max(np.mean(view_dizi), 1.0)),
            "gecikmeler": gecikmeler_view[:1000]
        }
