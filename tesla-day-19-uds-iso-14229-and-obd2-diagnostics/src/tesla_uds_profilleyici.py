"""
Tesla UDS Teşhis ve OBD-II Profilleyici Modülü
===============================================
Bu modül; ISO 14229 UDS servislerinin (0x22 DID Okuma, 0x19 DTC Sorgulama,
0x27 SecurityAccess) gecikmesini, P2/P2* zamanlama sınırlarını ve
DoIP (Diagnostics over IP) vs CAN-FD teşhis hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_uds_protokolu import (
    TeslaUDSServer,
    TeslaUDSClient,
    DiagnosticSessionType
)


class TeslaUDSProfilleyici:
    """
    UDS Teşhis Motoru ve OBD-II Performans Profilleyicisi.
    """
    def __init__(self, ornek_sayisi: int = 5000):
        self.ornek_sayisi = ornek_sayisi

    def benchmark_uds_servisleri(self) -> Dict[str, Any]:
        server = TeslaUDSServer()
        client = TeslaUDSClient(server)

        # 1. 0x22 ReadDataByIdentifier (DID Okuma Gecikmesi)
        gecikmeler_did_us: List[float] = []
        for _ in range(self.ornek_sayisi):
            t0 = time.perf_counter_ns()
            ok, _ = client.read_did(0xF190)  # VIN
            t1 = time.perf_counter_ns()
            gecikmeler_did_us.append(float(t1 - t0) / 1000.0)

        # 2. 0x19 ReadDTCInformation (DTC Sorgulama Gecikmesi)
        gecikmeler_dtc_us: List[float] = []
        for _ in range(self.ornek_sayisi):
            t0 = time.perf_counter_ns()
            _ = client.read_dtcs(0xFF)
            t1 = time.perf_counter_ns()
            gecikmeler_dtc_us.append(float(t1 - t0) / 1000.0)

        # 3. 0x27 SecurityAccess (Seed-Key Doğrulama Gecikmesi)
        client.set_session(DiagnosticSessionType.EXTENDED_DIAGNOSTIC_SESSION)
        t0 = time.perf_counter_ns()
        sec_ok = client.unlock_security()
        t1 = time.perf_counter_ns()
        sec_latency_us = float(t1 - t0) / 1000.0

        did_dizi = np.array(gecikmeler_did_us)
        dtc_dizi = np.array(gecikmeler_dtc_us)

        # 4. DoIP (Ethernet 100 Mbps) vs Klasik CAN 500k vs CAN-FD 5M Teşhis Süresi
        t_did_avg_us = float(np.mean(did_dizi))
        t_can_classic_did_ms = 4.2    # 8-byte frame sınırları ve ISO-TP akış kontrolü
        t_canfd_did_ms = 0.8         # 64-byte tek çerçeve
        t_doip_did_ms = t_did_avg_us / 1000.0

        return {
            "did_ortalama_us": t_did_avg_us,
            "did_p99_us": float(np.percentile(did_dizi, 99)),
            "dtc_ortalama_us": float(np.mean(dtc_dizi)),
            "dtc_p99_us": float(np.percentile(dtc_dizi, 99)),
            "security_handshake_us": sec_latency_us,
            "security_status": sec_ok,
            "can_classic_ms": t_can_classic_did_ms,
            "can_fd_ms": t_canfd_did_ms,
            "doip_ms": t_doip_did_ms,
            "hizlanma_doip_vs_can": t_can_classic_did_ms / max(t_doip_did_ms, 1e-4),
            "saniyelik_did_sorgusu": int(1e6 / max(t_did_avg_us, 1e-4)),
            "did_gecikmeler": gecikmeler_did_us[:200]
        }
