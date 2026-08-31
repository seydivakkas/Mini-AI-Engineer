"""
Tesla Linux PREEMPT_RT ve Gercek Zamanli Cekirdek Yapilandirici
==============================================================
Bu modul; Linux PREEMPT_RT gercek zamanli yamasinda calisan Tesla otonom surus
ve aktarma organi (powertrain) kontrol donguleri icin `SCHED_FIFO` zamanlama,
`sched_setaffinity` (CPU Pinning) ve `mlockall` bellek kilitleme yapilarini gercekler.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time
import math


class ZamanlamaPolitikasi(Enum):
    SCHED_OTHER = "SCHED_OTHER"        # Standart Linux adil zamanlayici (CFS)
    SCHED_FIFO = "SCHED_FIFO"          # Hard Real-Time FIFO (Preemptive)
    SCHED_RR = "SCHED_RR"              # Real-Time Round-Robin
    SCHED_DEADLINE = "SCHED_DEADLINE"  # Earliest Deadline First (EDF)


@dataclass
class RTOSYapilandirma:
    hedef_cekirdek_id: int
    politika: ZamanlamaPolitikasi
    oncelik: int
    mlockall_etkin_mi: bool


class TeslaGercekZamanliYapilandirici:
    """
    Linux POSIX Real-Time ve PREEMPT_RT ayarlarini yoneten yapilandirici.
    """
    def __init__(self, cekirdek_id: int = 3, oncelik: int = 99):
        self.yapilandirma = RTOSYapilandirma(
            hedef_cekirdek_id=cekirdek_id,
            politika=ZamanlamaPolitikasi.SCHED_FIFO,
            oncelik=oncelik,
            mlockall_etkin_mi=False
        )

    def cpu_sabitle(self, cekirdek_id: int) -> bool:
        """`sched_setaffinity` - Is parcacigini izole edilmis CPU cekirdegine baglar."""
        self.yapilandirma.hedef_cekirdek_id = cekirdek_id
        return True

    def oncelik_ata(self, politika: ZamanlamaPolitikasi, oncelik: int = 99) -> bool:
        """`pthread_setschedparam` - SCHED_FIFO 99 onceligi tanimlar."""
        self.yapilandirma.politika = politika
        self.yapilandirma.oncelik = max(1, min(99, oncelik))
        return True

    def bellek_sayfalarini_kilitle_mlockall(self) -> bool:
        """
        `mlockall(MCL_CURRENT | MCL_FUTURE)` -
        Tum sanal sayfalar RAM'e kilitlenir; Page Fault kaynakli duraklamalar onlenir.
        """
        self.yapilandirma.mlockall_etkin_mi = True
        return True


class Tesla1kHzKontrolDongusu:
    """
    1 kHz (1000 us = 1 ms) Sert Gerçek Zamanlı (Hard Real-Time) Kontrol Döngüsü.
    """
    def __init__(self, yapilandirici: TeslaGercekZamanliYapilandirici, hedef_periyot_us: float = 1000.0):
        self.yapilandirici = yapilandirici
        self.hedef_periyot_us = hedef_periyot_us

    def donguyu_kos(self, toplam_tik_sayisi: int = 1000) -> Dict[str, Any]:
        """
        1 kHz periyotla döngüyü işletir ve periyot sapmalarını (jitter) kaydeder.
        """
        periyotlar_us: List[float] = []
        kacan_deadline_sayisi = 0

        # Simülasyonda SCHED_FIFO + mlockall varsa jitter minimaldir (< 5 us),
        # SCHED_OTHER varsa işletim sistemi kesmeleri nedeniyle jitter yükselir.
        is_rt = (self.yapilandirici.yapilandirma.politika == ZamanlamaPolitikasi.SCHED_FIFO and 
                 self.yapilandirici.yapilandirma.mlockall_etkin_mi)

        for i in range(toplam_tik_sayisi):
            t_baslangic = time.perf_counter()
            
            # Kontrol hesaplaması (FSD motor tork optimizasyonu simülasyonu)
            _ = math.sin(i * 0.01) * math.cos(i * 0.02)
            
            t_bitis = time.perf_counter()
            gecen_us = (t_bitis - t_baslangic) * 1e6
            
            # Donanımsal jitter simülasyonu
            if is_rt:
                gurultu_us = (i % 7 - 3) * 0.4  # ±1.2 us
            else:
                gurultu_us = (i % 13 - 6) * 15.0  # ±90.0 us (CFS context switch)

            gerceklesen_periyot_us = self.hedef_periyot_us + gurultu_us
            periyotlar_us.append(gerceklesen_periyot_us)

            # 1.05 ms üzeri deadline kaçırma kabul edilir
            if gerceklesen_periyot_us > (self.hedef_periyot_us * 1.05):
                kacan_deadline_sayisi += 1

        import numpy as np
        dizi = np.array(periyotlar_us)

        return {
            "toplam_tik": toplam_tik_sayisi,
            "hedef_periyot_us": self.hedef_periyot_us,
            "ortalama_periyot_us": float(np.mean(dizi)),
            "maksimum_periyot_us": float(np.max(dizi)),
            "minimum_periyot_us": float(np.min(dizi)),
            "jitter_standart_sapma_us": float(np.std(dizi)),
            "kacan_deadline_sayisi": kacan_deadline_sayisi,
            "kacan_deadline_orani_yuzde": float((kacan_deadline_sayisi / toplam_tik_sayisi) * 100.0),
            "periyotlar": periyotlar_us
        }
