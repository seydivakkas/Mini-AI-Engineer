"""
Tesla Faz 3 Capstone Profilleyici Modülü
========================================
Bu modül; 0-120 km/h tam gaz ivmelenme, otoyol seyri, tek pedallı rejen duruşu
ve HVIL acil güvenlik kesme senaryolarını tek bir uçtan uca profil testinde birleştirir.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_faz3_capstone_bms_ve_cekis_cekirdegi import CapstonePowertrainCore


class TeslaCapstoneProfilleyici:
    """
    Faz 3 Büyük Capstone Güç Aktarma ve BMS Profilleyicisi.
    """
    def __init__(self, sim_adimlari: int = 1500):
        self.sim_adimlari = sim_adimlari

    def benchmark_capstone_surus_dongusu(self) -> Dict[str, Any]:
        powertrain = CapstonePowertrainCore()

        speed_history = []
        torque_history = []
        power_history = []
        voltage_history = []
        soc_history = []
        temp_history = []
        gecikmeler_step_us: List[float] = []

        dt = 0.01  # 10 ms (100 Hz)

        for step in range(self.sim_adimlari):
            # Senaryo Zaman Çizelgesi:
            # 0 - 600 adım (0-6 sn): %100 Gaz (0-120 km/h Ludicrous İvmelenme)
            # 600 - 1000 adım (6-10 sn): %25 Gaz (120 km/h Otoyol Seyri)
            # 1000 - 1400 adım (10-14 sn): %0 Gaz (Tek Pedallı Rejenerasyon ile Duruş)
            # 1400 - 1500 adım (14-15 sn): HVIL Kesilme Arıza Enjeksiyonu

            if step < 600:
                accel = 100.0
                target_v = 120.0
            elif step < 1000:
                accel = 25.0
                target_v = 120.0
            elif step < 1400:
                accel = 0.0
                target_v = 0.0
            else:
                # HVIL arızası enjekte edilir
                powertrain.hvil_closed = False
                accel = 50.0
                target_v = 100.0

            t0 = time.perf_counter_ns()
            out = powertrain.step_powertrain_cycle(
                accel_pedal_pct=accel,
                brake_pedal_pct=0.0,
                target_speed_kmh=target_v,
                dt_s=dt
            )
            t1 = time.perf_counter_ns()
            gecikmeler_step_us.append(float(t1 - t0) / 1000.0)

            speed_history.append(out["speed_kmh"])
            torque_history.append(out["torque_nm"])
            power_history.append(out["power_kw"])
            voltage_history.append(out.get("pack_voltage_v", 0.0))
            soc_history.append(out["soc_pct"])
            temp_history.append(out.get("pack_temp_c", 25.0))

        dizi = np.array(gecikmeler_step_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "capstone_step_ortalama_us": t_avg_us,
            "capstone_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_capstone_adimi": int(1e6 / max(t_avg_us, 1e-4)),
            "max_speed_kmh": float(np.max(speed_history)),
            "max_power_kw": float(np.max(power_history)),
            "max_regen_power_kw": float(abs(np.min(power_history))),
            "final_soc_pct": soc_history[-1],
            "speed": speed_history,
            "torque": torque_history,
            "power": power_history,
            "voltage": voltage_history,
            "soc": soc_history,
            "temp": temp_history,
            "capstone_gecikmeler": gecikmeler_step_us[:200]
        }
