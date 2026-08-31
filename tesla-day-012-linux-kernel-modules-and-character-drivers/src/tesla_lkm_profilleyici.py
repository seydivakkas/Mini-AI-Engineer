"""
Tesla Karakter Surucusu ve ioctl Profilleyicisi
===============================================
Bu modul; Linux kernel `ioctl` dogrudan aygit kontrolu ile Userspace
sysfs/metin tabanli dosya yazma arasindaki komut gecikmesini ve ASIL-D
guvenlik basarimini analiz eder.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_karakter_surucusu import (
    TeslaTorkKarakterAygiti,
    TeslaTorkPaketi,
    IOCTL_TESLA_TORK_YAZ,
    ASIL_D_GUVENLIK_ANAHTARI
)


class TeslaLKMProfilleyici:
    """
    Karakter surucusu ve ioctl profilleyici.
    """
    def __init__(self, komut_sayisi: int = 5000):
        self.komut_sayisi = komut_sayisi

    def benchmark_ioctl_vs_sysfs_gecikmesi(self) -> Dict[str, Any]:
        aygit = TeslaTorkKarakterAygiti()
        aygit.open()

        gecerli_paket = TeslaTorkPaketi(
            guvenlik_anahtari=ASIL_D_GUVENLIK_ANAHTARI,
            hedef_tork_nm=450.0,
            rejenerasyon_etkin_mi=False
        )
        paket_baytlari = gecerli_paket.to_bytes()

        # 1. Kernel ioctl Doğrudan Komut İletimi
        gecikmeler_ioctl_us: List[float] = []
        for _ in range(self.komut_sayisi):
            t0 = time.perf_counter_ns()
            kod, _ = aygit.unlocked_ioctl(IOCTL_TESLA_TORK_YAZ, paket_baytlari)
            t1 = time.perf_counter_ns()
            gecikmeler_ioctl_us.append(float(t1 - t0) / 1000.0)

        # 2. Sysfs Metin Ayrıştırma Simülasyonu ("echo 450.0 > /sys/class/tesla/torque")
        gecikmeler_sysfs_us: List[float] = []
        for _ in range(self.komut_sayisi):
            t0 = time.perf_counter_ns()
            metin = "450.0\n"
            ayristirilan = float(metin.strip())
            _ = ayristirilan * 1.0
            t1 = time.perf_counter_ns()
            gecikmeler_sysfs_us.append(float(t1 - t0) / 1000.0)

        ioctl_dizi = np.array(gecikmeler_ioctl_us)
        sysfs_dizi = np.array(gecikmeler_sysfs_us)

        aygit.release()

        return {
            "ioctl_ortalama_us": float(np.mean(ioctl_dizi)),
            "ioctl_p99_us": float(np.percentile(ioctl_dizi, 99)),
            "sysfs_ortalama_us": float(np.mean(sysfs_dizi)),
            "sysfs_p99_us": float(np.percentile(sysfs_dizi, 99)),
            "hizlanma_orani": float(np.mean(sysfs_dizi) / max(np.mean(ioctl_dizi), 1e-4)),
            "saniyelik_tork_komut_kapasitesi": float(1e6 / max(np.mean(ioctl_dizi), 1e-3)),
            "gecikmeler_ioctl": gecikmeler_ioctl_us[:300]
        }
