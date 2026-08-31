"""
Tesla IRQ Profilleyici Modulu
=============================
Bu modul; Top-Half HardIRQ gecikmesi ile monolitik bloklayici kesme isleyici
arasindaki gecikme farkini ve kesme firtinasi filtreleme basarimini analiz eder.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_kesme_yoneticisi import (
    TeslaTopHalfHardIRQ,
    TeslaBottomHalfThreadedIRQ,
    TeslaKesmeFirtinasiOnleyici,
    TeslaKesmeYonetimSistemi
)


class TeslaIRQProfilleyici:
    """
    Linux Kesme Yonetim Sistemi Performans Profilleyicisi.
    """
    def __init__(self, ornek_sayisi: int = 5000):
        self.ornek_sayisi = ornek_sayisi

    def benchmark_tophalf_vs_monolitik(self) -> Dict[str, Any]:
        top_half = TeslaTopHalfHardIRQ()
        bottom_half = TeslaBottomHalfThreadedIRQ()

        # 1. Top-Half HardIRQ Gecikmesi (Sadece ACK)
        gecikmeler_tophalf_us: List[float] = []
        for _ in range(self.ornek_sayisi):
            t0 = time.perf_counter_ns()
            _ = top_half.kesme_isle()
            t1 = time.perf_counter_ns()
            gecikmeler_tophalf_us.append(float(t1 - t0) / 1000.0)

        # 2. Monolitik Bloklayıcı Kesme (Top-Half içinde tüm radar hesaplamasını yapma hatası)
        gecikmeler_monolitik_us: List[float] = []
        for _ in range(self.ornek_sayisi):
            t0 = time.perf_counter_ns()
            _ = top_half.kesme_isle()
            _ = bottom_half.radar_nokta_bulutu_isle(25.0, -15.0)
            t1 = time.perf_counter_ns()
            gecikmeler_monolitik_us.append(float(t1 - t0) / 1000.0)

        th_dizi = np.array(gecikmeler_tophalf_us)
        mono_dizi = np.array(gecikmeler_monolitik_us)

        # 3. Kesme Fırtınası Filtreleme Testi (50,000 istek)
        firtina_filtresi = TeslaKesmeFirtinasiOnleyici(maks_irq_hizi_sn=10000)
        kabul_sayisi = 0
        red_sayisi = 0
        for _ in range(20000):
            if firtina_filtresi.kesme_kabul_edilebilir_mi():
                kabul_sayisi += 1
            else:
                red_sayisi += 1

        return {
            "tophalf_ortalama_us": float(np.mean(th_dizi)),
            "tophalf_p99_us": float(np.percentile(th_dizi, 99)),
            "monolitik_ortalama_us": float(np.mean(mono_dizi)),
            "hizlanma_orani": float(np.mean(mono_dizi) / max(np.mean(th_dizi), 1e-4)),
            "firtina_kabul_orani": float(kabul_sayisi / (kabul_sayisi + red_sayisi) * 100),
            "firtina_red_orani": float(red_sayisi / (kabul_sayisi + red_sayisi) * 100),
            "gecikmeler_tophalf": gecikmeler_tophalf_us[:200]
        }
