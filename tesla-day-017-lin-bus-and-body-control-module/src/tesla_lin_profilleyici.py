"""
Tesla LIN Profilleyici Modulu
=============================
Bu modul; LIN veri yolu cerceve iletim surelerini, PID parite hesaplama
gecikmesini ve cizelgeleme tablosu zamanlama hassasiyetini analiz eder.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_lin_protokolu import (
    TeslaLINSlaveBCM,
    TeslaLINMasterCizelgeleyici,
    pid_hesapla,
    pid_dogrula,
    gelismis_checksum_hesapla
)


class TeslaLINProfilleyici:
    """
    LIN Veri Yolu ve BCM Performans Profilleyicisi.
    """
    def __init__(self, ornek_sayisi: int = 5000):
        self.ornek_sayisi = ornek_sayisi

    def benchmark_lin_performansi(self) -> Dict[str, Any]:
        slave = TeslaLINSlaveBCM()
        master = TeslaLINMasterCizelgeleyici(slave)

        # 1. PID ve Checksum Hesaplama Gecikmesi
        gecikmeler_pid_us: List[float] = []
        for _ in range(self.ornek_sayisi):
            t0 = time.perf_counter_ns()
            pid = pid_hesapla(0x32)
            _ = pid_dogrula(pid)
            _ = gelismis_checksum_hesapla(pid, b'\x50')
            t1 = time.perf_counter_ns()
            gecikmeler_pid_us.append(float(t1 - t0) / 1000.0)

        # 2. LIN 19.2 kbps İletim Süresi (Teorik: Break 13b + Sync 10b + PID 10b + 2B Veri 20b + Csum 10b = 63 bit)
        bit_sayisi = 63
        t_lin_19k2_ms = (bit_sayisi / 19200.0) * 1000.0  # ~3.28 ms
        t_lin_9k6_ms = (bit_sayisi / 9600.0) * 1000.0   # ~6.56 ms

        # 3. Çizelge Tablosu İletim Simülasyonu
        cizelge_sonuclari = []
        for gorev in master.cizelge_tablosu:
            msg = master.cerceve_gonder(gorev["frame_id"], gorev["veri"])
            islem = slave.lin_mesaj_isle(msg)
            cizelge_sonuclari.append({
                "gorev": gorev["isim"],
                "pid": hex(msg.pid),
                "aygit": islem.get("aygit"),
                "deger": islem.get("yeni_deger")
            })

        pid_dizi = np.array(gecikmeler_pid_us)

        return {
            "pid_ortalama_us": float(np.mean(pid_dizi)),
            "pid_p99_us": float(np.percentile(pid_dizi, 99)),
            "lin_19k2_sure_ms": t_lin_19k2_ms,
            "lin_9k6_sure_ms": t_lin_9k6_ms,
            "cizelge_sonuclari": cizelge_sonuclari,
            "gecikmeler_pid": gecikmeler_pid_us[:200]
        }
