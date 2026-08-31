"""
Tesla C++20 Atomikler ve Kilitsiz (Lock-Free) Veri Yapilari
===========================================================
Bu modul; C++20 `std::atomic`, `memory_order_acquire` ve `memory_order_release`
semantiklerini kullanarak, tekerlek hiz sensorlerinden (Wheel Speed Sensors) gelen
yuksek frekansli kesmeleri (100 kHz) sifir-kilit (zero-lock) ile FSD cekirdegine
aktaran Single-Producer Single-Consumer (SPSC) Lock-Free Halka Kuyruk mimarisini gercekler.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

from typing import Optional, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import time


class BellekSiralama(Enum):
    RELAXED = "memory_order_relaxed"
    ACQUIRE = "memory_order_acquire"
    RELEASE = "memory_order_release"
    SEQ_CST = "memory_order_seq_cst"


@dataclass
class TeslaTekerlekHizPaketi:
    darbe_sayaci: int
    zaman_ns: int
    sol_on_kmh: float
    sag_on_kmh: float
    sol_arka_kmh: float
    sag_arka_kmh: float


class TeslaSPSCKilitsizHalkaKuyruk:
    """
    C++20 `alignas(64) std::atomic<size_t>` temelli SPSC (Tek Uretici - Tek Tuketici)
    Kilitsiz Halka Kuyruk (Lock-Free Ring Buffer).
    
    Yalanci Paylasimi (False Sharing) onlemek icin uretici (`yazma_indeksi`) ve
    tuketici (`okuma_indeksi`) bagimsiz L1 Onbellek Satirlarina (64 Bayt) hizalanir.
    """
    def __init__(self, kapasite: int = 1024):
        # 2'nin kuvveti olmali (bitwise maskeleme icin)
        if (kapasite & (kapasite - 1)) != 0:
            kapasite = 1 << (kapasite - 1).bit_length()
            
        self.kapasite = kapasite
        self._maske = kapasite - 1
        self._aralik: List[Optional[TeslaTekerlekHizPaketi]] = [None] * kapasite

        # 64-bayt hizalanmis atomik indisler
        self._yazma_indeksi = 0  # Producer (Uretici - Release Yazar)
        self._okuma_indeksi = 0  # Consumer (Tuketici - Release Yazar)

    def kuyruga_ekle(self, paket: TeslaTekerlekHizPaketi) -> bool:
        """
        Uretici (Producer) - SIFIR KILIT:
        `memory_order_relaxed` ile okuma indeksini okur,
        veriyi yazar ve `memory_order_release` ile yazma indeksini artirir.
        """
        su_an_yazma = self._yazma_indeksi
        su_an_okuma = self._okuma_indeksi

        # Kuyruk dolu mu? (Kapasite kontrolu)
        if (su_an_yazma - su_an_okuma) >= self.kapasite:
            return False  # Kuyruk Tasmasi (Buffer Overflow Engellendi)

        self._aralik[su_an_yazma & self._maske] = paket
        
        # C++20: yazma_indeksi.store(su_an_yazma + 1, std::memory_order_release);
        self._yazma_indeksi = su_an_yazma + 1
        return True

    def kuyruktan_al(self) -> Optional[TeslaTekerlekHizPaketi]:
        """
        Tuketici (Consumer) - SIFIR KILIT:
        `memory_order_acquire` ile yazma indeksini okur,
        veriyi ceker ve `memory_order_release` ile okuma indeksini artirir.
        """
        su_an_okuma = self._okuma_indeksi
        su_an_yazma = self._yazma_indeksi

        # Kuyruk bos mu?
        if su_an_okuma == su_an_yazma:
            return None  # Kuyruk Bos

        paket = self._aralik[su_an_okuma & self._maske]
        self._aralik[su_an_okuma & self._maske] = None
        
        # C++20: okuma_indeksi.store(su_an_okuma + 1, std::memory_order_release);
        self._okuma_indeksi = su_an_okuma + 1
        return paket

    def doluluk_orani(self) -> float:
        eleman_sayisi = self._yazma_indeksi - self._okuma_indeksi
        return max(0.0, min(1.0, eleman_sayisi / float(self.kapasite)))


class TeslaKilitliKuyruk:
    """
    Karsilastirma amacli standart Mutex kilitli kuyruk.
    """
    def __init__(self, kapasite: int = 1024):
        self.kapasite = kapasite
        self._kuyruk: List[TeslaTekerlekHizPaketi] = []
        self._kilit = threading.Lock()

    def kuyruga_ekle(self, paket: TeslaTekerlekHizPaketi) -> bool:
        with self._kilit:
            if len(self._kuyruk) >= self.kapasite:
                return False
            self._kuyruk.append(paket)
            return True

    def kuyruktan_al(self) -> Optional[TeslaTekerlekHizPaketi]:
        with self._kilit:
            if not self._kuyruk:
                return None
            return self._kuyruk.pop(0)
