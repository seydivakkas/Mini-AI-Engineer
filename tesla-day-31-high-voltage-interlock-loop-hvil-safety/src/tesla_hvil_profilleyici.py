"""
Tesla HVIL ve Güvenlik Profilleyici Modülü
==========================================
Bu modül; HVIL açık devre acil durdurma gecikmesini, Precharge şarj eğrisini,
Pyrofuse patlatma süresini ve 1 kHz ASIL-D güvenlik döngü hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_hvil_ve_guvenlik_sistemi import (
    TeslaHVILSafetyManager,
    HighVoltageSystemState,
    ContactorState,
    HVILStatus
)


class TeslaHVILProfilleyici:
    """
    Yüksek Gerilim Güvenlik Sistemi Performans Profilleyicisi.
    """
    def __init__(self, sim_ms: int = 500):
        self.sim_ms = sim_ms

    def benchmark_hvil_guvenlik(self) -> Dict[str, Any]:
        safety_mgr = TeslaHVILSafetyManager()
        state = HighVoltageSystemState(v_battery_dc=400.0, v_inverter_link=0.0, r_isolation_kohm=600.0)

        v_link_history = []
        state_history = []
        gecikmeler_step_us: List[float] = []

        # 1. Normal Precharge ve Başlatma (0 - 300 ms)
        for t in range(300):
            t0 = time.perf_counter_ns()
            out = safety_mgr.execute_safety_cycle(state, dt_ms=1.0)
            t1 = time.perf_counter_ns()
            gecikmeler_step_us.append(float(t1 - t0) / 1000.0)

            v_link_history.append(out["v_link"])
            state_history.append(out["contactor_state"])

        precharge_success_time_ms = len([s for s in state_history if s == "PRECHARGE_ACTIVE"])

        # 2. 301. ms'de HVIL Konnektör Kopması (Fault Injection)
        state.hvil_signal_valid = False
        hvil_out = safety_mgr.execute_safety_cycle(state, dt_ms=1.0)
        v_link_history.append(hvil_out["v_link"])
        state_history.append(hvil_out["contactor_state"])

        # 3. Kaza Sinyali ve Pyrofuse Patlatma Testi
        state_crash = HighVoltageSystemState(v_battery_dc=400.0, crash_signal_rcm=True)
        crash_out = safety_mgr.execute_safety_cycle(state_crash, dt_ms=1.0)

        dizi = np.array(gecikmeler_step_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "hvil_step_ortalama_us": t_avg_us,
            "hvil_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_guvenlik_adimi": int(1e6 / max(t_avg_us, 1e-4)),
            "precharge_time_ms": precharge_success_time_ms,
            "hvil_shutdown_status": hvil_out["contactor_state"],
            "pyrofuse_blown": not state_crash.pyrofuse_intact,
            "v_link_history": v_link_history,
            "state_history": state_history,
            "hvil_gecikmeler": gecikmeler_step_us[:200]
        }
