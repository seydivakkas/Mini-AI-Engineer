"""
Tesla SocketCAN ve Filtreleme Profilleyicisi
============================================
Bu modul; Linux Kernel seviyesinde SocketCAN donanimsal maskeleme filtresi ile
kullanici alaninda (Userspace) yazilimsal filtreleme arasindaki gecikme,
CPU ek yuku ve paket gecis verimini olcer.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_socketcan_arayuzu import (
    TeslaCANFrame,
    TeslaCANFiltresi,
    TeslaSocketCANArayuzu,
    TeslaVCanAgSimulatoru
)


class TeslaSocketCANProfilleyici:
    """
    SocketCAN performans analizoru.
    """
    def __init__(self, paket_sayisi: int = 10000):
        self.paket_sayisi = paket_sayisi

    def benchmark_kernel_vs_userspace_filtreleme(self) -> Dict[str, Any]:
        """
        Kernel seviyesi SocketCAN mask filtreleme ile Userspace döngü filtreleme karşılaştırması.
        """
        ornek_frame_100 = TeslaCANFrame(can_id=0x100, can_dlc=8, data=b'\x01\x02\x03\x04\x05\x06\x07\x08')
        ornek_frame_300 = TeslaCANFrame(can_id=0x300, can_dlc=8, data=b'\xAA\xBB\xCC\xDD\xEE\xFF\x00\x11')

        # 1. Kernel Seviyesi Filtreleme (Sadece 0x100 kabul et)
        kernel_soket = TeslaSocketCANArayuzu("vcan0")
        kernel_soket.baglan()
        kernel_soket.filtre_ekle(can_id=0x100, can_mask=0x7FF)

        gecikmeler_kernel: List[float] = []
        for i in range(self.paket_sayisi):
            f = ornek_frame_100 if (i % 2 == 0) else ornek_frame_300
            t0 = time.perf_counter_ns()
            kernel_soket.kernel_filtresinden_gecir_ve_kabul_et(f)
            t1 = time.perf_counter_ns()
            gecikmeler_kernel.append(float(t1 - t0))

        # 2. Userspace Filtreleme (Her paketi userspace'e alıp if ile kontrol etme)
        userspace_soket = TeslaSocketCANArayuzu("vcan0")
        userspace_soket.baglan()

        gecikmeler_userspace: List[float] = []
        for i in range(self.paket_sayisi):
            f = ornek_frame_100 if (i % 2 == 0) else ornek_frame_300
            t0 = time.perf_counter_ns()
            userspace_soket.gelen_kuyruk.append(f)
            # Userspace filter
            alinan = userspace_soket.gelen_kuyruk.pop()
            if alinan.can_id == 0x100:
                pass
            t1 = time.perf_counter_ns()
            gecikmeler_userspace.append(float(t1 - t0))

        k_dizi = np.array(gecikmeler_kernel)
        u_dizi = np.array(gecikmeler_userspace)

        return {
            "kernel_ort_ns": float(np.mean(k_dizi)),
            "kernel_p99_ns": float(np.percentile(k_dizi, 99)),
            "userspace_ort_ns": float(np.mean(u_dizi)),
            "userspace_p99_ns": float(np.percentile(u_dizi, 99)),
            "hizlanma_orani": float(np.mean(u_dizi) / max(np.mean(k_dizi), 1e-3)),
            "saniyede_frame_kapasitesi": float(1e9 / max(np.mean(k_dizi), 1.0)),
            "gecikmeler": gecikmeler_kernel[:1000]
        }
