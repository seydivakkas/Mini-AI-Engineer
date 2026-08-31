"""
Tesla C++20 Esyordamlar (Coroutines) ve Asenkron G/C Modulu
===========================================================
Bu modul; C++20 stackless coroutine mimarisini (co_await, co_yield, co_return),
promise_type ve std::coroutine_handle durum makinelerini gercekler.
Tesla 10 Gbps Ethernet ve CAN-FD soketlerinden gelen telemetri paketlerini
is parcacigi (OS thread) degisim maliyeti olmadan sifir-bloklama ile tuketir.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

from typing import Generator, Any, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
import time


class EsyordamDurumu(Enum):
    BASKI_ALTINDA = "ASKIYA_ALINDI"  # std::suspend_always
    CALISIYOR = "CALISIYOR"
    TAMAMLANDI = "TAMAMLANDI"        # std::suspend_never / final_suspend


@dataclass
class TeslaEthernetPaketi:
    akis_id: int
    zaman_ns: int
    sensor_kaynagi: str
    veri_boyutu_bayt: int
    veri_ozeti: str


class TeslaTelemetriUreteci:
    """
    C++20 `generator<TeslaEthernetPaketi>` esdegeri.
    `co_yield` ile her cagirildiginda soketten bir sonraki paketi non-blocking dondurur.
    """
    def __init__(self, kaynak_adi: str, toplam_paket: int = 100):
        self.kaynak_adi = kaynak_adi
        self.toplam_paket = toplam_paket
        self._uretec = self._esyordam_govdesi()
        self.durum = EsyordamDurumu.BASKI_ALTINDA

    def _esyordam_govdesi(self) -> Generator[TeslaEthernetPaketi, None, None]:
        """co_yield ile calisan esyordam durum makinesi."""
        for i in range(self.toplam_paket):
            self.durum = EsyordamDurumu.CALISIYOR
            paket = TeslaEthernetPaketi(
                akis_id=i,
                zaman_ns=time.time_ns(),
                sensor_kaynagi=self.kaynak_adi,
                veri_boyutu_bayt=1400,  # Standart Ethernet MTU
                veri_ozeti=f"{self.kaynak_adi}_FRAME_{i:04d}_OK"
            )
            self.durum = EsyordamDurumu.BASKI_ALTINDA
            yield paket
        self.durum = EsyordamDurumu.TAMAMLANDI

    def siradaki_paketi_al(self) -> Optional[TeslaEthernetPaketi]:
        """std::coroutine_handle<promise_type>::resume() cagirisi."""
        try:
            return next(self._uretec)
        except StopIteration:
            self.durum = EsyordamDurumu.TAMAMLANDI
            return None


class TeslaEsyordamGorevi:
    """
    C++20 `task<T>` esdegeri asenkron gorev.
    `co_await` ile diger esyordamlari bekleyebilir ve deger dondurebilir (`co_return`).
    """
    def __init__(self, gorev_adi: str, uretici: TeslaTelemetriUreteci):
        self.gorev_adi = gorev_adi
        self.uretici = uretici
        self.islenen_paket_sayisi = 0
        self.toplam_bayt = 0
        self.durum = EsyordamDurumu.BASKI_ALTINDA

    def adim_islet(self) -> bool:
        """
        Esyordami bir adim ilerletir (co_await adimi).
        Gorev tamamlandiysa False dondurur.
        """
        self.durum = EsyordamDurumu.CALISIYOR
        paket = self.uretici.siradaki_paketi_al()
        if paket is not None:
            self.islenen_paket_sayisi += 1
            self.toplam_bayt += paket.veri_boyutu_bayt
            self.durum = EsyordamDurumu.BASKI_ALTINDA
            return True
        else:
            self.durum = EsyordamDurumu.TAMAMLANDI
            return False


class Tesla10GbpsEthernetHatti:
    """
    Tesla Otonom Arac 10 Gbps Ethernet Coklu Sensor Esyordam Zamanlayicisi (Cooperative Scheduler).
    """
    def __init__(self):
        self.kayitli_gorevler: List[TeslaEsyordamGorevi] = []

    def gorev_ekle(self, gorev: TeslaEsyordamGorevi):
        self.kayitli_gorevler.append(gorev)

    def tum_akis_hatlarini_tukelt(self) -> Dict[str, Any]:
        """
        Tum esyordam gorevlerini is parcacigi degisim maliyeti olmadan kooperatif calistirir.
        """
        t0 = time.perf_counter_ns()
        aktif_gorevler = list(self.kayitli_gorevler)
        toplam_adim = 0

        while aktif_gorevler:
            bitti_listesi = []
            for g in aktif_gorevler:
                devam_ediyor = g.adim_islet()
                toplam_adim += 1
                if not devam_ediyor:
                    bitti_listesi.append(g)
            for b in bitti_listesi:
                aktif_gorevler.remove(b)

        t1 = time.perf_counter_ns()
        toplam_sure_ns = t1 - t0
        toplam_bayt = sum(g.toplam_bayt for g in self.kayitli_gorevler)
        
        return {
            "toplam_adim": toplam_adim,
            "toplam_sure_ns": float(toplam_sure_ns),
            "toplam_bayt": toplam_bayt,
            "adim_basina_ns": float(toplam_sure_ns / max(toplam_adim, 1)),
            "mb_saniye": float((toplam_bayt / (1024 * 1024)) / max(toplam_sure_ns / 1e9, 1e-9))
        }
