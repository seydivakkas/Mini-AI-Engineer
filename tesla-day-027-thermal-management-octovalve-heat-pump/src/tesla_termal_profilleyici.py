"""
Tesla Octovalve Termal Profilleyici Modülü
==========================================
Bu modül; Supercharger öncesi batarya ön koşullandırma (Preconditioning)
süresini, Isı Pompası COP verimini ve PTC dirençli ısıtıcıya göre enerji tasarrufunu profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_octovalve_termal_yonetim import (
    TeslaOctovalveController,
    VehicleThermalState,
    OctovalveMode
)


class TeslaTermalProfilleyici:
    """
    Termal Yönetim ve Isı Pompası Performans Profilleyicisi.
    """
    def __init__(self, sim_saniye: int = 1800):  # 30 dakika
        self.sim_saniye = sim_saniye

    def benchmark_termal_sistem(self) -> Dict[str, Any]:
        ctrl = TeslaOctovalveController(target_battery_temp_c=45.0, target_cabin_temp_c=22.0)
        state_hp = VehicleThermalState(temp_battery_c=5.0, temp_cabin_c=10.0, temp_ambient_c=0.0)

        # 1. Isı Pompası & Octovalve ile Ön Isıtma (Preconditioning)
        batt_temp_hp = []
        cabin_temp_hp = []
        power_hp_w = []
        gecikmeler_step_us: List[float] = []

        for s in range(self.sim_saniye):
            mode = ctrl.determine_mode(state_hp, supercharge_target_set=True)

            t0 = time.perf_counter_ns()
            out = ctrl.step(state_hp, mode, dt_s=1.0)
            t1 = time.perf_counter_ns()
            gecikmeler_step_us.append(float(t1 - t0) / 1000.0)

            batt_temp_hp.append(out["temp_battery_c"])
            cabin_temp_hp.append(out["temp_cabin_c"])
            power_hp_w.append(out["compressor_power_w"])

        # 2. Geleneksel Dirençli PTC Isıtıcı Simülasyonu (COP = 1.0)
        # Aynı termal gücü üretmek için kompresör gücünün 3.2 katı elektrik harcar!
        ptc_energy_kwh = (np.sum(power_hp_w) * 3.2) / (3600.0 * 1000.0)
        hp_energy_kwh = np.sum(power_hp_w) / (3600.0 * 1000.0)
        energy_saved_pct = ((ptc_energy_kwh - hp_energy_kwh) / ptc_energy_kwh) * 100.0

        step_dizi = np.array(gecikmeler_step_us)
        t_step_avg_us = float(np.mean(step_dizi))

        return {
            "termal_step_ortalama_us": t_step_avg_us,
            "termal_step_p99_us": float(np.percentile(step_dizi, 99)),
            "saniyelik_termal_adimi": int(1e6 / max(t_step_avg_us, 1e-4)),
            "final_battery_temp_c": batt_temp_hp[-1],
            "final_cabin_temp_c": cabin_temp_hp[-1],
            "hp_energy_kwh": hp_energy_kwh,
            "ptc_energy_kwh": ptc_energy_kwh,
            "energy_saved_pct": energy_saved_pct,
            "batt_temp_history": batt_temp_hp,
            "cabin_temp_history": cabin_temp_hp
        }
