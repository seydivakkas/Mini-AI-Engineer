"""
Tesla epoll ve Olay Tabanli Coklama Profilleyicisi
==================================================
Bu modul; Linux `epoll` ($O(1)$) ile geleneksel `select`/`poll` ($O(N)$)
coklayicilarinin soket sayisi arttikca gecikme ve olceklenme basarimini olcer.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_epoll_reaktoru import (
    TeslaEpollOlayReaktoru,
    EpollTetiklemeModu,
    EpollOlayTipi
)


class TeslaEpollProfilleyici:
    """
    epoll vs select/poll O(1) vs O(N) profilleyici.
    """
    def __init__(self, fd_sayilari: List[int] = None):
        self.fd_sayilari = fd_sayilari or [50, 200, 500, 1000, 2000]

    def benchmark_olceklenme_analizi(self) -> Dict[str, Any]:
        epoll_gecikmeleri_us: List[float] = []
        select_gecikmeleri_us: List[float] = []

        for n in self.fd_sayilari:
            # 1. epoll Reaktörü Kur
            reaktor = TeslaEpollOlayReaktoru()
            for fd in range(n):
                reaktor.epoll_ctl_ekle(fd_id=fd, olay_maskesi=EpollOlayTipi.EPOLLIN, kullanici_verisi=f"soket_{fd}")

            # Sadece 3 sokete veri geldi
            reaktor.veri_geldi_sinyali(fd_id=5, bayt_sayisi=1024)
            reaktor.veri_geldi_sinyali(fd_id=n // 2, bayt_sayisi=512)
            reaktor.veri_geldi_sinyali(fd_id=n - 1, bayt_sayisi=256)

            # epoll_wait ölçümü (O(1))
            t0 = time.perf_counter_ns()
            _ = reaktor.epoll_wait(maks_olay=16)
            t1 = time.perf_counter_ns()
            epoll_gecikmeleri_us.append(float(t1 - t0) / 1000.0)

            # select simülasyonu (Tüm N soketi tek tek dolaşan O(N) döngü)
            t0_sel = time.perf_counter_ns()
            hazirlar_select = []
            for fd in range(n):
                if fd in [5, n // 2, n - 1]:
                    hazirlar_select.append(fd)
            t1_sel = time.perf_counter_ns()
            select_gecikmeleri_us.append(float(t1_sel - t0_sel) / 1000.0)

        epoll_dizi = np.array(epoll_gecikmeleri_us)
        select_dizi = np.array(select_gecikmeleri_us)

        return {
            "fd_sayilari": self.fd_sayilari,
            "epoll_gecikmeleri_us": epoll_gecikmeleri_us,
            "select_gecikmeleri_us": select_gecikmeleri_us,
            "maksimum_hizlanma": float(select_gecikmeleri_us[-1] / max(epoll_gecikmeleri_us[-1], 1e-4)),
            "epoll_ortalama_us": float(np.mean(epoll_dizi)),
            "select_ortalama_us": float(np.mean(select_dizi))
        }
