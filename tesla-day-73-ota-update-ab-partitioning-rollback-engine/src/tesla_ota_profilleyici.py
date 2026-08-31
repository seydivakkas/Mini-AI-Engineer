"""
Tesla OTA Profilleyici Modülü
=============================
Bu modül; A/B slot geçiş süresini, otomatik geri alma (Rollback) gecikmesini
ve OTA durum makinesi çözüm hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_ota_ab_slot_yonetici import OTABootSlotManager


class TeslaOTAProfilleyici:
    """
    Tesla OTA ve Rollback Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_ota_rollback(self) -> Dict[str, Any]:
        mgr = OTABootSlotManager()

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            mgr_inst = OTABootSlotManager()
            t0 = time.perf_counter_ns()
            ciktilar = mgr_inst.simulate_corrupted_ota_rollback()
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "rollback_step_ortalama_us": t_avg_us,
            "rollback_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_gecis_hacmi": int(1e6 / max(t_avg_us, 1e-4)),
            "final_slot": ciktilar["final_active_slot"],
            "final_version": ciktilar["final_version"],
            "rollback_success": ciktilar["rollback_success"],
            "events": ciktilar["event_history"],
            "gecikmeler": gecikmeler_us[:200]
        }
