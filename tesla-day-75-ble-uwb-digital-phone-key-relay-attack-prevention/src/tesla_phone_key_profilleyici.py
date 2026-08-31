"""
Tesla Phone Key Profilleyici Modülü
===================================
Bu modül; UWB ToF mesafe hesaplama hızını, BLE+UWB füzyon süresini ve
röle saldırısı algılama gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_phone_key_uwb_dogrulayici import TeslaPhoneKeyUWBValidator


class TeslaPhoneKeyProfilleyici:
    """
    Tesla Phone Key Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_phone_key(self) -> Dict[str, Any]:
        validator = TeslaPhoneKeyUWBValidator()

        gecikmeler_us: List[float] = []
        ciktilar_normal = None
        ciktilar_attack = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar_normal = validator.evaluate_phone_key_unlock(ble_rssi_dbm=-62.0, uwb_tof_ns=4.5)
            ciktilar_attack = validator.evaluate_phone_key_unlock(ble_rssi_dbm=-50.0, uwb_tof_ns=35.0)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi)) / 2.0  # 2 evaluations per iter

        return {
            "tof_check_ortalama_us": t_avg_us,
            "tof_check_p99_us": float(np.percentile(dizi, 99)) / 2.0,
            "saniyelik_kilit_kontrolu": int(1e6 / max(t_avg_us, 1e-4)),
            "normal_dist": ciktilar_normal["calculated_distance_m"],
            "normal_unlock": ciktilar_normal["door_unlocked"],
            "attack_dist": ciktilar_attack["calculated_distance_m"],
            "attack_detected": ciktilar_attack["relay_attack_detected"],
            "gecikmeler": gecikmeler_us[:200]
        }
