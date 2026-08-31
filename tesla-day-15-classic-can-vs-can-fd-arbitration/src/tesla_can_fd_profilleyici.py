"""
Tesla CAN vs CAN-FD Profilleyici Modulu
=======================================
Bu modul; Klasik CAN ile CAN-FD arasindaki iletim surelerini,
efektif bant genisligini ve arbitrasyon oncelik siralamasini analiz eder.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_can_fd_protokolu import (
    TeslaKlasikCANFrame,
    TeslaCANFDFrame,
    TeslaCANArbitrasyonSimulasyonu
)


class TeslaCANFDProfilleyici:
    """
    CAN 2.0 vs CAN-FD Veri Yolu Performans ve Arbitrasyon Analizoru.
    """
    def __init__(self):
        pass

    def benchmark_can_vs_can_fd(self) -> Dict[str, Any]:
        # 1. Klasik CAN Çerçevesi (8 byte @ 500 kbps)
        klasik_frame = TeslaKlasikCANFrame(can_id=0x100, veri=b'\x00' * 8)
        t_klasik_us = klasik_frame.iletim_suresi_us_hesapla()
        bant_genisligi_klasik_kbps = (8 * 8) / (t_klasik_us * 1e-6) / 1000.0  # kbps

        # 2. CAN-FD Çerçevesi (64 byte @ 500kbps Arb + 5Mbps Data)
        fd_frame = TeslaCANFDFrame(can_id=0x100, veri=b'\x00' * 64, brs_aktif_mi=True)
        t_fd_us = fd_frame.iletim_suresi_us_hesapla()
        bant_genisligi_fd_kbps = (64 * 8) / (t_fd_us * 1e-6) / 1000.0  # kbps

        # 3. Arbitrasyon Senaryosu (3 Düğüm Aynı Anda Veri Yolunda)
        sim = TeslaCANArbitrasyonSimulasyonu()
        sim.mesaj_ekle("Tesla_Infotainment", 0x380, b'MEDIA_PLAY', "Medya Bilgi Ekranı (Düşük Öncelik)")
        sim.mesaj_ekle("Tesla_Fren_Modulu", 0x010, b'BRAKE_NOW', "ASIL-D Acil Fren (En Yüksek Öncelik)")
        sim.mesaj_ekle("Tesla_Surucu_Motor", 0x120, b'TORQUE_CMD', "Tork Talebi (Orta Öncelik)")
        
        arb_sonuc = sim.arbitrasyon_yaristir()

        return {
            "klasik_sure_us": t_klasik_us,
            "can_fd_sure_us": t_fd_us,
            "klasik_bant_kbps": bant_genisligi_klasik_kbps,
            "can_fd_bant_kbps": bant_genisligi_fd_kbps,
            "bant_genisligi_carpani": bant_genisligi_fd_kbps / bant_genisligi_klasik_kbps,
            "arbitrasyon_kazanan": arb_sonuc["kazanan"],
            "arbitrasyon_elenenler": arb_sonuc["elenenler"]
        }
