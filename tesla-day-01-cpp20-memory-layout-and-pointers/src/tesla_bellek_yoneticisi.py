"""
Tesla Bellek Yoneticisi (Tesla C++20 Memory Layout & Zero-Allocation Pool)
==========================================================================
Bu modul, Tesla otonom arac gomulu yazilim cekirdeginde kullanilan deterministik,
64-byte cache line hizali ve sifir dinamik tahsis (zero-allocation) bellek yonetim
mimarilerini gercekler.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

from dataclasses import dataclass
import time
import struct
import numpy as np
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class TeslaTelemetriPaketi:
    """
    Tesla CAN-FD ve Otonom Surus Sensor Veri Paketi.
    C++20 struct alignas(64) standardina uygun, 64 baytlik cache-line dostu yapi.
    """
    paket_id: int
    zaman_damgasi_ns: int
    can_id: int
    direksiyon_acisi_rad: float
    arac_hizi_kmh: float
    batarya_gerilimi_v: float
    motor_torku_nm: float
    fren_basinci_bar: float
    kontrol_checksum: int

    def baytlara_donustur(self) -> bytes:
        """Paketi 64 baytlik ikili (binary) formata paketler."""
        # 8 bayt int, 8 bayt int, 4 bayt int, 5 x 8 bayt double, 4 bayt int = 64 bayt
        return struct.pack(
            "=QQI5dI",
            self.paket_id,
            self.zaman_damgasi_ns,
            self.can_id,
            self.direksiyon_acisi_rad,
            self.arac_hizi_kmh,
            self.batarya_gerilimi_v,
            self.motor_torku_nm,
            self.fren_basinci_bar,
            self.kontrol_checksum
        )

    @classmethod
    def baytlardan_coz(cls, veri: bytes) -> 'TeslaTelemetriPaketi':
        """64 baytlik ikili formattan nesne olusturur."""
        degerler = struct.unpack("=QQI5dI", veri[:64])
        return cls(
            paket_id=degerler[0],
            zaman_damgasi_ns=degerler[1],
            can_id=degerler[2],
            direksiyon_acisi_rad=degerler[3],
            arac_hizi_kmh=degerler[4],
            batarya_gerilimi_v=degerler[5],
            motor_torku_nm=degerler[6],
            fren_basinci_bar=degerler[7],
            kontrol_checksum=degerler[8]
        )


class CacheHizaliBellekHavuzu:
    """
    Tesla Otonom Surus Bilgisayari (HW3/HW4) icin 64-Bayt Hizali Bellek Havuzu.
    Sifir dinamik heap tahsisi (Zero Dynamic Heap Allocation) prensibiyle onceden
    tahsis edilmis bitisik bellek blogunu yonetir.
    """
    def __init__(self, blok_sayisi: int = 1024, blok_boyutu: int = 64, hizalama_bayt: int = 64):
        self.blok_sayisi = blok_sayisi
        self.blok_boyutu = blok_boyutu
        self.hizalama_bayt = hizalama_bayt
        
        # Bitisik bellek alani (Pre-allocated monolithic buffer)
        self.toplam_kapasite_bayt = blok_sayisi * blok_boyutu
        self.bellek_alani = bytearray(self.toplam_kapasite_bayt)
        
        # Bos blok takip listesi (Free list)
        self.bos_indeksler: List[int] = list(range(blok_sayisi))
        self.dolu_indeksler: set = set()
        
        # Istatistiki metrikler
        self.toplam_tahsis_sayisi = 0
        self.toplam_serbest_birakma_sayisi = 0
        self.maksimum_kullanim = 0

    def tahsis_et(self, veri: bytes) -> Optional[int]:
        """
        Havuzdan O(1) karmasiklikta 64-bayt hizali bellek blogu tahsis eder.
        Dinamik malloc/new cagrilarini engelleyerek deterministik zamanlama saglar.
        """
        if not self.bos_indeksler:
            return None  # Havuz dolu (OOM onleme)

        blok_idx = self.bos_indeksler.pop()
        self.dolu_indeksler.add(blok_idx)
        
        baslangic = blok_idx * self.blok_boyutu
        bitis = baslangic + min(len(veri), self.blok_boyutu)
        
        self.bellek_alani[baslangic:bitis] = veri[:self.blok_boyutu]
        
        self.toplam_tahsis_sayisi += 1
        self.maksimum_kullanim = max(self.maksimum_kullanim, len(self.dolu_indeksler))
        return blok_idx

    def serbest_birak(self, blok_idx: int) -> bool:
        """
        Tahsis edilmis bellek blogunu O(1) surede havuza iade eder.
        """
        if blok_idx not in self.dolu_indeksler:
            return False
        
        self.dolu_indeksler.remove(blok_idx)
        self.bos_indeksler.append(blok_idx)
        self.toplam_serbest_birakma_sayisi += 1
        return True

    def blok_oku(self, blok_idx: int) -> Optional[bytes]:
        """Belirtilen bloktaki veriyi kopyalamadan okur."""
        if blok_idx not in self.dolu_indeksler:
            return None
        baslangic = blok_idx * self.blok_boyutu
        return bytes(self.bellek_alani[baslangic:baslangic + self.blok_boyutu])

    def doluluk_orani(self) -> float:
        """Havuz doluluk oranini dondurur."""
        return len(self.dolu_indeksler) / self.blok_sayisi

    def parcalanma_indeksi(self) -> float:
        """
        Bellek parcalanma indeksi (Fragmentation Index).
        Sabit boyutlu blok havuzunda sifir parcalanma (0.0) hedeflenir.
        """
        return 0.0  # Sabit bloklu havuzda harici parcalanma (external fragmentation) sifirdir.


class SifirTahsilliHalkaKuyruk:
    """
    Lock-Free Single-Producer Single-Consumer (SPSC) Halka Kuyruk (Ring Buffer).
    CAN-FD ve IMU sensor telemetrisi icin gercek zamanli deterministik FIFO.
    """
    def __init__(self, kapasite: int = 512):
        self.kapasite = kapasite
        self.tampon: List[Optional[TeslaTelemetriPaketi]] = [None] * kapasite
        self.yazma_imleci = 0
        self.okuma_imleci = 0
        self.eleman_sayisi = 0
        self.toplam_eklenen = 0
        self.toplam_tasan = 0

    def ekle(self, paket: TeslaTelemetriPaketi) -> bool:
        """Kuyruga yeni telemetri paketi ekler. Tasma durumunda en eski veriyi ezmez."""
        if self.eleman_sayisi >= self.kapasite:
            self.toplam_tasan += 1
            return False
        
        self.tampon[self.yazma_imleci] = paket
        self.yazma_imleci = (self.yazma_imleci + 1) % self.kapasite
        self.eleman_sayisi += 1
        self.toplam_eklenen += 1
        return True

    def cikar(self) -> Optional[TeslaTelemetriPaketi]:
        """Kuyruktan siradaki telemetri paketini FIFO sirasiyla ceker."""
        if self.eleman_sayisi == 0:
            return None
        
        paket = self.tampon[self.okuma_imleci]
        self.tampon[self.okuma_imleci] = None
        self.okuma_imleci = (self.okuma_imleci + 1) % self.kapasite
        self.eleman_sayisi -= 1
        return paket

    def bos_mu(self) -> bool:
        return self.eleman_sayisi == 0

    def dolu_mu(self) -> bool:
        return self.eleman_sayisi >= self.kapasite

    def doluluk_orani(self) -> float:
        return self.eleman_sayisi / self.kapasite
