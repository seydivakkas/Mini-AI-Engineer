"""
Tesla RTOS ve Zamanlayici Profilleyicisi
========================================
Bu modul; Linux PREEMPT_RT `SCHED_FIFO` ve CPU Pinning ile standart `SCHED_OTHER`
arasindaki determinizm, periyot dalgalanmasi (jitter) ve deadline kacirma
oranlarini karsilastirmali olarak analiz eder.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

from typing import Dict, Any
from src.tesla_rtos_cekirdek import (
    TeslaGercekZamanliYapilandirici,
    Tesla1kHzKontrolDongusu,
    ZamanlamaPolitikasi
)


class TeslaRTOSProfilleyici:
    """
    PREEMPT_RT vs Non-RT profilleyici.
    """
    def __init__(self, tik_sayisi: int = 1000):
        self.tik_sayisi = tik_sayisi

    def benchmark_rt_vs_non_rt(self) -> Dict[str, Any]:
        """
        Hard Real-Time (SCHED_FIFO + mlockall + Core 3 Pinning) ile Standart Linux karsilastirmasi.
        """
        # 1. Hard Real-Time Döngü
        rt_yapici = TeslaGercekZamanliYapilandirici(cekirdek_id=3, oncelik=99)
        rt_yapici.bellek_sayfalarini_kilitle_mlockall()
        rt_dongu = Tesla1kHzKontrolDongusu(rt_yapici, hedef_periyot_us=1000.0)
        rt_sonuclari = rt_dongu.donguyu_kos(toplam_tik_sayisi=self.tik_sayisi)

        # 2. Standart Non-RT Döngü (SCHED_OTHER, mlockall yok)
        non_rt_yapici = TeslaGercekZamanliYapilandirici(cekirdek_id=0, oncelik=0)
        non_rt_yapici.oncelik_ata(ZamanlamaPolitikasi.SCHED_OTHER, oncelik=0)
        non_rt_dongu = Tesla1kHzKontrolDongusu(non_rt_yapici, hedef_periyot_us=1000.0)
        non_rt_sonuclari = non_rt_dongu.donguyu_kos(toplam_tik_sayisi=self.tik_sayisi)

        return {
            "rt_ortalama_us": rt_sonuclari["ortalama_periyot_us"],
            "rt_jitter_us": rt_sonuclari["jitter_standart_sapma_us"],
            "rt_maksimum_us": rt_sonuclari["maksimum_periyot_us"],
            "rt_kacan_deadline": rt_sonuclari["kacan_deadline_sayisi"],
            "rt_kacan_yuzde": rt_sonuclari["kacan_deadline_orani_yuzde"],
            "rt_periyotlar": rt_sonuclari["periyotlar"][:300],

            "non_rt_ortalama_us": non_rt_sonuclari["ortalama_periyot_us"],
            "non_rt_jitter_us": non_rt_sonuclari["jitter_standart_sapma_us"],
            "non_rt_maksimum_us": non_rt_sonuclari["maksimum_periyot_us"],
            "non_rt_kacan_deadline": non_rt_sonuclari["kacan_deadline_sayisi"],
            "non_rt_kacan_yuzde": non_rt_sonuclari["kacan_deadline_orani_yuzde"],
            "non_rt_periyotlar": non_rt_sonuclari["periyotlar"][:300],

            "jitter_iyilesme_orani": float(non_rt_sonuclari["jitter_standart_sapma_us"] / max(rt_sonuclari["jitter_standart_sapma_us"], 1e-3))
        }
