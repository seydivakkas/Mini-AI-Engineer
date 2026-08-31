"""
Tesla Faz 2 Capstone Profilleyici Modülü
========================================
Bu modül; Merkezi Ağ Gateway yönlendirme gecikmesini, çoklu veri yolu
(Multi-Bus) işlem kapasitesini ve UDS teşhis sorgulama başarımını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import struct
import numpy as np
from typing import Dict, Any, List
from src.tesla_faz2_capstone_gateway import TeslaCentralGateway


class TeslaCapstoneProfilleyici:
    """
    Merkezi Araç Gateway ve Teşhis Performans Profilleyicisi.
    """
    def __init__(self, ornek_sayisi: int = 5000):
        self.ornek_sayisi = ornek_sayisi

    def benchmark_gateway_pipeline(self) -> Dict[str, Any]:
        gw = TeslaCentralGateway()

        # Örnek CAN-FD Paketleri
        payload_powertrain = struct.pack(">HHh2x", 4000, 1500, 345)  # 400.0 V, 150.0 A, 34.5 °C
        payload_chassis = struct.pack(">HH4x", 2400, 1850)          # 120.0 km/h, +5.0 deg
        payload_lin = bytes([0x01, 0x00])

        gecikmeler_gw_us: List[float] = []
        for _ in range(self.ornek_sayisi):
            t0 = time.perf_counter_ns()
            gw.decode_canfd_powertrain(0x301, payload_powertrain)
            gw.decode_canfd_chassis(0x12F, payload_chassis)
            gw.decode_lin_bcm(0x24, payload_lin)
            _ = gw.handle_uds_request(bytes([0x22, 0x01, 0x04]))  # Read Power
            t1 = time.perf_counter_ns()
            gecikmeler_gw_us.append(float(t1 - t0) / 1000.0)

        gw_dizi = np.array(gecikmeler_gw_us)
        t_gw_avg_us = float(np.mean(gw_dizi))

        return {
            "gateway_ortalama_us": t_gw_avg_us,
            "gateway_p99_us": float(np.percentile(gw_dizi, 99)),
            "saniyelik_gateway_hacmi": int(1e6 / max(t_gw_avg_us, 1e-4)),
            "islenen_toplam_frame": gw.processed_frames_count,
            "hesaplanan_guc_kw": gw.state.power_kw,
            "arac_hizi_kmh": gw.state.vehicle_speed_kmh,
            "gateway_gecikmeler": gecikmeler_gw_us[:200]
        }
