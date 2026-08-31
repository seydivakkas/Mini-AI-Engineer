"""
Tesla SOME/IP Profilleyici Modulu
=================================
Bu modul; SOME/IP RPC cagri gecikmesini, baslik paketleme/cozme
maliyetini ve Automotive Ethernet (1 Gbps) ile CAN-FD arasindaki hiz farkini analiz eder.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_someip_protokolu import (
    TeslaSOMEIPServer,
    TeslaSOMEIPClient,
    TeslaSOMEIPHeader,
    TeslaSOMEIPPaket,
    SOMEIPMessageType,
    SOMEIPReturnCode
)


class TeslaSOMEIPProfilleyici:
    """
    SOME/IP ve Otomotiv Ethernet Performans Profilleyicisi.
    """
    def __init__(self, ornek_sayisi: int = 5000):
        self.ornek_sayisi = ornek_sayisi

    def benchmark_someip_rpc(self) -> Dict[str, Any]:
        server = TeslaSOMEIPServer(service_id=0x1234)
        client = TeslaSOMEIPClient(client_id=0x0042)

        # 1. SOME/IP RPC Çağrı Gecikmesi (Header + Payload + Decode + Server Execution + Encode)
        gecikmeler_rpc_us: List[float] = []
        for _ in range(self.ornek_sayisi):
            t0 = time.perf_counter_ns()
            ok, _ = client.rpc_hedef_hiz_cagir(server, 120.0)
            t1 = time.perf_counter_ns()
            gecikmeler_rpc_us.append(float(t1 - t0) / 1000.0)

        # 2. CAN-FD Üzerinden Eşdeğer RPC İletim Süresi (CAN-FD BRS 5 Mbps: ~172 µs)
        t_can_fd_rpc_us = 172.0
        # 3. Automotive Ethernet 1000BASE-T1 (1 Gbps) İletim Süresi (~2.1 µs)
        t_eth_rpc_us = float(np.mean(gecikmeler_rpc_us))

        rpc_dizi = np.array(gecikmeler_rpc_us)

        return {
            "someip_ortalama_us": t_eth_rpc_us,
            "someip_p99_us": float(np.percentile(rpc_dizi, 99)),
            "can_fd_rpc_us": t_can_fd_rpc_us,
            "hizlanma_carpani": t_can_fd_rpc_us / max(t_eth_rpc_us, 1e-4),
            "saniyelik_rpc_kapasitesi": int(1e6 / max(t_eth_rpc_us, 1e-4)),
            "gecikmeler_rpc": gecikmeler_rpc_us[:200]
        }
