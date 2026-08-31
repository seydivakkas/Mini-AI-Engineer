"""
Tesla ISO 15118 Profilleyici Modülü
===================================
Bu modül; ISO 15118 Tak-Çalıştır el sıkışma hızını ve V2G mesajlaşma
çerçevesi üretim gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_nacs_iso15118_motor import TeslaNACSISO15118Engine


class TeslaISO15118Profilleyici:
    """
    Tesla ISO 15118 ve NACS Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_plug_and_charge(self) -> Dict[str, Any]:
        engine = TeslaNACSISO15118Engine()

        gecikmeler_us: List[float] = []
        ciktilar_auth = None
        ciktilar_v2g = None

        for _ in range(self.iterations):
            eng_inst = TeslaNACSISO15118Engine()
            t0 = time.perf_counter_ns()
            _ = eng_inst.handle_plug_connection()
            ciktilar_auth = eng_inst.verify_iso15118_contract(
                vehicle_vin="5YJ3E1EB8NF123456",
                contract_token="CONTRACT_TESLA_VIP_9988",
                oem_signature="SIG_ECDSA_TESLA_CA_AUTH_KEY_2026"
            )
            ciktilar_v2g = eng_inst.create_v2g_charge_loop_message(
                target_voltage_v=400.0,
                max_current_a=500.0,
                soc_pct=35.0
            )
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "pnc_step_ortalama_us": t_avg_us,
            "pnc_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_mesaj_kapasitesi": int(1e6 / max(t_avg_us, 1e-4)),
            "vin": ciktilar_auth["vin"],
            "auth_status": ciktilar_auth["authorization_status"],
            "cp_state": ciktilar_auth["cp_state"],
            "power_kw": ciktilar_v2g["charging_power_kw"],
            "gecikmeler": gecikmeler_us[:200]
        }
