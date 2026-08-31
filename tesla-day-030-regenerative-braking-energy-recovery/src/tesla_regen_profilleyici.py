"""
Tesla Rejeneratif Frenleme Profilleyici Modülü
===============================================
Bu modül; 80 km/h'den 0 km/h'ye tek pedallı duruş testini, geri kazanılan
enerjiyi (Wh) ve sıcak/soğuk batarya rejen sınırlarını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_rejeneratif_fren_yonetimi import (
    TeslaRegenerativeBrakeController,
    VehicleDynamicsState,
    StoppingMode
)


class TeslaRegenProfilleyici:
    """
    Rejeneratif Frenleme ve Enerji Geri Kazanım Profilleyicisi.
    """
    def __init__(self, init_speed_kmh: float = 80.0):
        self.init_speed = init_speed_kmh

    def benchmark_rejenerasyon(self) -> Dict[str, Any]:
        ctrl = TeslaRegenerativeBrakeController()

        # 1. Normal Batarya (25°C, %70 SoC) Tek Pedallı Durma Simülasyonu
        state_warm = VehicleDynamicsState(
            speed_kmh=self.init_speed,
            accel_pedal_pct=0.0,
            brake_pedal_pct=0.0,
            battery_soc=0.70,
            battery_temp_c=25.0
        )

        speed_warm_list = []
        regen_torque_warm = []
        regen_power_warm = []
        gecikmeler_step_us: List[float] = []

        dt = 0.01  # 10 ms (100 Hz)
        for _ in range(3000):  # Maksimum 30 saniye
            t0 = time.perf_counter_ns()
            out = ctrl.step_torque_blending(state_warm, dt_s=dt)
            t1 = time.perf_counter_ns()
            gecikmeler_step_us.append(float(t1 - t0) / 1000.0)

            speed_warm_list.append(out["speed_kmh"])
            regen_torque_warm.append(out["regen_torque_nm"])
            regen_power_warm.append(out["regen_power_kw"])

            if out["speed_kmh"] <= 0.0:
                break

        # Geri kazanılan toplam enerji (Wh) = integral(P * dt) / 3600
        recovered_energy_wh = float(np.sum(regen_power_warm) * 1000.0 * dt) / 3600.0

        # 2. Soğuk Batarya (-5°C) Kısıtlamalı Simülasyon
        state_cold = VehicleDynamicsState(
            speed_kmh=self.init_speed,
            accel_pedal_pct=0.0,
            brake_pedal_pct=0.0,
            battery_soc=0.70,
            battery_temp_c=-5.0  # Dondurucu soğuk!
        )
        speed_cold_list = []
        for _ in range(3000):
            out_c = ctrl.step_torque_blending(state_cold, dt_s=dt)
            speed_cold_list.append(out_c["speed_kmh"])
            if out_c["speed_kmh"] <= 0.0:
                break

        step_dizi = np.array(gecikmeler_step_us)
        t_step_avg_us = float(np.mean(step_dizi))

        return {
            "regen_step_ortalama_us": t_step_avg_us,
            "regen_step_p99_us": float(np.percentile(step_dizi, 99)),
            "saniyelik_regen_adimi": int(1e6 / max(t_step_avg_us, 1e-4)),
            "stopping_time_warm_s": len(speed_warm_list) * dt,
            "recovered_energy_wh": recovered_energy_wh,
            "max_regen_power_kw": float(np.max(regen_power_warm)),
            "speed_warm": speed_warm_list,
            "speed_cold": speed_cold_list,
            "regen_torque": regen_torque_warm,
            "regen_power": regen_power_warm,
            "regen_gecikmeler": gecikmeler_step_us[:200]
        }
