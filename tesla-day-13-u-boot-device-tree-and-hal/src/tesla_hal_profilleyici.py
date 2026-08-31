"""
Tesla HAL ve Device Tree Profilleyicisi
=======================================
Bu modul; Device Tree tabanli HAL erisim gecikmesi ile dinamik aygit tarama
arasindaki hiz farkini ve U-Boot Fast Boot acilis asamalarini analiz eder.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_device_tree_ve_hal import (
    TeslaDeviceTreeParser,
    TeslaUBootAcilisSekansi,
    TeslaDonanimSoyutlamaKatmani
)


class TeslaHALProfilleyici:
    """
    Device Tree HAL performans analizoru.
    """
    def __init__(self, ornek_sayisi: int = 5000):
        self.ornek_sayisi = ornek_sayisi

    def benchmark_hal_vs_dinamik_tarama(self) -> Dict[str, Any]:
        dt = TeslaDeviceTreeParser()
        dt.standart_tesla_hw4_agacini_yukle()
        hal = TeslaDonanimSoyutlamaKatmani(dt)

        # 1. Device Tree HAL Doğrudan Sensör Okuma
        gecikmeler_hal_us: List[float] = []
        for _ in range(self.ornek_sayisi):
            t0 = time.perf_counter_ns()
            _ = hal.i2c_sicaklik_oku(0x48)
            _ = hal.spi_imu_oku()
            t1 = time.perf_counter_ns()
            gecikmeler_hal_us.append(float(t1 - t0) / 1000.0)

        # 2. Dinamik Runtime Aygıt Tarama Simülasyonu
        gecikmeler_tarama_us: List[float] = []
        for _ in range(self.ornek_sayisi):
            t0 = time.perf_counter_ns()
            # Her seferinde I2C bus adreslerini 0x03'ten 0x77'ye tarama
            bulunanlar = []
            for addr in range(3, 120):
                if addr in [0x48, 0x49]:
                    bulunanlar.append(addr)
            t1 = time.perf_counter_ns()
            gecikmeler_tarama_us.append(float(t1 - t0) / 1000.0)

        hal_dizi = np.array(gecikmeler_hal_us)
        tarama_dizi = np.array(gecikmeler_tarama_us)

        uboot = TeslaUBootAcilisSekansi()
        boot_sonuclari = uboot.acilisi_gerceklestir()

        return {
            "hal_ortalama_us": float(np.mean(hal_dizi)),
            "tarama_ortalama_us": float(np.mean(tarama_dizi)),
            "hizlanma_orani": float(np.mean(tarama_dizi) / max(np.mean(hal_dizi), 1e-4)),
            "boot_asamalari": boot_sonuclari["asamalar"],
            "toplam_boot_ms": boot_sonuclari["toplam_acilis_suresi_ms"],
            "gecikmeler_hal": gecikmeler_hal_us[:200]
        }
