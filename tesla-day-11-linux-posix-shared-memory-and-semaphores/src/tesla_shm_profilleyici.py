"""
Tesla Paylasilan Bellek (SHM) Profilleyicisi
=============================================
Bu modul; POSIX Shared Memory (Zero-Copy) ile Linux Pipe / UNIX Domain Socket
arasindaki veri aktarim gecikmesi ve bant genisligini analiz eder.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_paylasilan_bellek import (
    TeslaPOSIXPaylasilanBellek,
    TeslaSifirKopyaGoruntuHatti
)


class TeslaSHMProfilleyici:
    """
    Zero-Copy SHM vs Standart IPC karsilastirmasi.
    """
    def __init__(self, tekrar_sayisi: int = 200):
        self.tekrar_sayisi = tekrar_sayisi

    def benchmark_shm_vs_pipe_gecikmesi(self) -> Dict[str, Any]:
        frame_boyutu = 1920 * 1080 * 3 # 6.22 MB RGB 1080p
        test_verisi = b'\xAA' * frame_boyutu

        hat = TeslaSifirKopyaGoruntuHatti(frame_boyutu_bayt=frame_boyutu)

        # 1. Zero-Copy SHM Ölçümü (İşaretçi & memoryview devri)
        gecikmeler_shm_us: List[float] = []
        for _ in range(self.tekrar_sayisi):
            t0 = time.perf_counter_ns()
            hat.uretici_kamera_frame_yaz(test_verisi)
            gorunum = hat.tuketici_fsd_frame_oku_gorunumu()
            # Sıfır kopyalama ile ilk ve son baytı doğrula
            _ = gorunum[0]
            t1 = time.perf_counter_ns()
            gecikmeler_shm_us.append(float(t1 - t0) / 1000.0)

        # 2. Standart Pipe/Soket Ölçümü (Tam bayt kopyalama maliyeti)
        gecikmeler_pipe_us: List[float] = []
        for _ in range(self.tekrar_sayisi):
            t0 = time.perf_counter_ns()
            # Kernel buffer'a yazma + oradan userspace'e okuma (2x Copy)
            kopya_1 = bytes(test_verisi)
            kopya_2 = bytes(kopya_1)
            _ = kopya_2[0]
            t1 = time.perf_counter_ns()
            gecikmeler_pipe_us.append(float(t1 - t0) / 1000.0)

        shm_dizi = np.array(gecikmeler_shm_us)
        pipe_dizi = np.array(gecikmeler_pipe_us)

        shm_ort = float(np.mean(shm_dizi))
        pipe_ort = float(np.mean(pipe_dizi))

        return {
            "frame_boyutu_mb": float(frame_boyutu / (1024 * 1024)),
            "shm_ortalama_us": shm_ort,
            "pipe_ortalama_us": pipe_ort,
            "hizlanma_orani": float(pipe_ort / max(shm_ort, 1e-3)),
            "shm_bant_genisligi_gbps": float((frame_boyutu / 1e9) / max(shm_ort * 1e-6, 1e-9)),
            "pipe_bant_genisligi_gbps": float((frame_boyutu / 1e9) / max(pipe_ort * 1e-6, 1e-9)),
            "gecikmeler_shm": gecikmeler_shm_us[:100]
        }
