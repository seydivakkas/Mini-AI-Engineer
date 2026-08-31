"""
Tesla Eszamanlilik ve Kilitsiz Kuyruk Profilleyicisi
====================================================
Bu modul; Lock-Free SPSC Halka Kuyruk ile Mutex Kilitli Kuyruk arasindaki
gecikme, jitter ve islem hacmi farkini karsilastirmali olarak olcer.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_kilitsiz_kuyruk import (
    TeslaTekerlekHizPaketi,
    TeslaSPSCKilitsizHalkaKuyruk,
    TeslaKilitliKuyruk
)


class TeslaEsZamanlilikProfilleyici:
    """
    Lock-Free SPSC vs Mutex Locked kuyruk performans analizoru.
    """
    def __init__(self, islem_sayisi: int = 50000):
        self.islem_sayisi = islem_sayisi

    def benchmark_spsc_vs_kilitli(self) -> Dict[str, Any]:
        """
        Lock-free SPSC ile standart Mutex kilitli kuyruk karsilastirmasi.
        """
        ornek_paket = TeslaTekerlekHizPaketi(
            darbe_sayaci=100,
            zaman_ns=time.time_ns(),
            sol_on_kmh=105.4,
            sag_on_kmh=105.2,
            sol_arka_kmh=104.9,
            sag_arka_kmh=105.0
        )

        # 1. Lock-Free SPSC Kuyruk
        spsc = TeslaSPSCKilitsizHalkaKuyruk(kapasite=4096)
        spsc_gecikmeler: List[float] = []

        for _ in range(self.islem_sayisi):
            t0 = time.perf_counter_ns()
            spsc.kuyruga_ekle(ornek_paket)
            spsc.kuyruktan_al()
            t1 = time.perf_counter_ns()
            spsc_gecikmeler.append(float(t1 - t0))

        # 2. Mutex Kilitli Kuyruk
        kilitli = TeslaKilitliKuyruk(kapasite=4096)
        kilitli_gecikmeler: List[float] = []

        for _ in range(self.islem_sayisi // 5):
            t0 = time.perf_counter_ns()
            kilitli.kuyruga_ekle(ornek_paket)
            kilitli.kuyruktan_al()
            t1 = time.perf_counter_ns()
            kilitli_gecikmeler.append(float(t1 - t0))

        spsc_dizi = np.array(spsc_gecikmeler)
        kilitli_dizi = np.array(kilitli_gecikmeler)

        return {
            "spsc_ort_ns": float(np.mean(spsc_dizi)),
            "spsc_p99_ns": float(np.percentile(spsc_dizi, 99)),
            "spsc_jitter_ns": float(np.std(spsc_dizi)),
            "spsc_milyon_islem_sn": float(1e9 / max(np.mean(spsc_dizi), 1.0) / 1e6),
            "kilitli_ort_ns": float(np.mean(kilitli_dizi)),
            "kilitli_p99_ns": float(np.percentile(kilitli_dizi, 99)),
            "kilitli_jitter_ns": float(np.std(kilitli_dizi)),
            "kilitli_milyon_islem_sn": float(1e9 / max(np.mean(kilitli_dizi), 1.0) / 1e6),
            "hizlanma_orani": float(np.mean(kilitli_dizi) / max(np.mean(spsc_dizi), 1e-3)),
            "spsc_gecikmeler": spsc_gecikmeler[:1000]
        }
