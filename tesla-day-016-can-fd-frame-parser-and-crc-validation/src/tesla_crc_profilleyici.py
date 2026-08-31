"""
Tesla CAN-FD CRC Profilleyici Modulu
====================================
Bu modul; CRC-17 ve CRC-21 hesaplama gecikmelerini, bozuk bit tespit
basarisini ve saniyelik cerceve ayristirma hizini analiz eder.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_can_fd_parser import (
    TeslaCANFDFrameParser,
    hesapla_crc17,
    hesapla_crc21
)


class TeslaCRCProfilleyici:
    """
    CAN-FD CRC ve Ayrıştırma Performans Profilleyicisi.
    """
    def __init__(self, ornek_sayisi: int = 5000):
        self.ornek_sayisi = ornek_sayisi

    def benchmark_crc_ve_ayristirma(self) -> Dict[str, Any]:
        parser = TeslaCANFDFrameParser()
        veri_16b = b'BATTERY_TEMP_OK!'  # 16 byte
        veri_64b = b'X' * 64              # 64 byte

        # 1. CRC-17 Hesaplama Gecikmesi (16 byte)
        gecikmeler_crc17_us: List[float] = []
        for _ in range(self.ornek_sayisi):
            t0 = time.perf_counter_ns()
            _ = hesapla_crc17(veri_16b)
            t1 = time.perf_counter_ns()
            gecikmeler_crc17_us.append(float(t1 - t0) / 1000.0)

        # 2. CRC-21 Hesaplama Gecikmesi (64 byte)
        gecikmeler_crc21_us: List[float] = []
        for _ in range(self.ornek_sayisi):
            t0 = time.perf_counter_ns()
            _ = hesapla_crc21(veri_64b)
            t1 = time.perf_counter_ns()
            gecikmeler_crc21_us.append(float(t1 - t0) / 1000.0)

        # 3. Bozuk Bit Tespit Testi (Bit-Flip Enjeksiyonu)
        paket_temiz = parser.cerceve_serilestir(0x100, veri_64b)
        
        # 1-bit bozulmuş paket
        paket_bozuk_1bit = bytearray(paket_temiz)
        paket_bozuk_1bit[10] ^= 0x01  # Bit flip
        
        # 2-bit bozulmuş paket
        paket_bozuk_2bit = bytearray(paket_temiz)
        paket_bozuk_2bit[15] ^= 0x80
        paket_bozuk_2bit[20] ^= 0x02

        sonuc_temiz = parser.cerceve_ayristir(bytes(paket_temiz))
        sonuc_bozuk1 = parser.cerceve_ayristir(bytes(paket_bozuk_1bit))
        sonuc_bozuk2 = parser.cerceve_ayristir(bytes(paket_bozuk_2bit))

        dizi17 = np.array(gecikmeler_crc17_us)
        dizi21 = np.array(gecikmeler_crc21_us)

        return {
            "crc17_ortalama_us": float(np.mean(dizi17)),
            "crc21_ortalama_us": float(np.mean(dizi21)),
            "crc21_p99_us": float(np.percentile(dizi21, 99)),
            "saniyede_islenen_cerceve": int(1e6 / max(np.mean(dizi21), 1e-4)),
            "temiz_paket_gecerli_mi": sonuc_temiz.gecerli_mi,
            "bozuk1_reddedildi_mi": not sonuc_bozuk1.gecerli_mi,
            "bozuk2_reddedildi_mi": not sonuc_bozuk2.gecerli_mi,
            "gecikmeler_crc21": gecikmeler_crc21_us[:200]
        }
