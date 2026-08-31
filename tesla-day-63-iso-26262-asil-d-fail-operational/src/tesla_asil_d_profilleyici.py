"""
Tesla ASIL-D Profilleyici Modülü
================================
Bu modül; ISO 26262 ASIL-D çift kanal kontrol hızını, debouncing arıza
takip süresini ve Fail-Operational durum geçiş gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_asil_d_guvenlik_kalkani import TeslaASILDSafetyGuard


class TeslaASILDProfilleyici:
    """
    ASIL-D Güvenlik Kalkanı Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_safety_guard(self) -> Dict[str, Any]:
        guard = TeslaASILDSafetyGuard(fault_debounce_threshold=3)

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = guard.process_safety_cycle(
                torque_ch1_nm=2.1,
                torque_ch2_nm=2.3,
                speed_ch1_mps=25.0,
                speed_ch2_mps=25.1
            )
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "safety_step_ortalama_us": t_avg_us,
            "safety_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_guvenlik_dongusu": int(1e6 / max(t_avg_us, 1e-4)),
            "state": ciktilar["safety_state"],
            "torque_diff": ciktilar["torque_diff_nm"],
            "speed_diff": ciktilar["speed_diff_mps"],
            "action": ciktilar["mrm_action"],
            "is_safe": ciktilar["is_safe"],
            "is_drive_allowed": ciktilar["is_drive_allowed"],
            "gecikmeler": gecikmeler_us[:200]
        }
