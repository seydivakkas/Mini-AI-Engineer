"""
Tesla D-Bus Profilleyici Modülü
===============================
Bu modül; D-Bus IPC RPC metod çağrısı gecikmesini,
sinyal yayınlama hızını ve IPC mesaj kuyruğu verimini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_dbus_bcm_yonetici import TeslaDBusBodyController, LightMode


class TeslaDBusProfilleyici:
    """
    Tesla D-Bus BCM Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_dbus_service(self) -> Dict[str, Any]:
        service = TeslaDBusBodyController()

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = service.simulate_ui_interaction_batch()
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))
        per_call_us = t_avg_us / 200.0  # 200 RPC calls per batch

        return {
            "dbus_call_ortalama_us": per_call_us,
            "dbus_batch_ortalama_us": t_avg_us,
            "dbus_batch_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_rpc_kapasitesi": int(1e6 / max(per_call_us, 1e-4)),
            "processed": ciktilar["processed_calls"],
            "total_signals": ciktilar["total_signals_emitted"],
            "lights": ciktilar["lights_mode"],
            "charge_port": ciktilar["charge_port_open"],
            "gecikmeler": gecikmeler_us[:200]
        }
