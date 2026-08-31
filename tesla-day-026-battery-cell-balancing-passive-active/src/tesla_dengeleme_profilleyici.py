"""
Tesla Hücre Dengeleme Profilleyici Modülü
==========================================
Bu modül; Pasif ve Aktif dengeleme algoritmalarının dengeleme hızını,
ısı kayıplarını ve 96S batarya modülündeki voltaj uyumunu profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_hucre_dengeleme_kontrolcusu import (
    BatteryCell,
    TeslaBalancingController,
    BalancingStrategy
)


class TeslaDengelemeProfilleyici:
    """
    Pasif ve Aktif Dengeleme Performans Profilleyicisi.
    """
    def __init__(self, num_cells: int = 12):
        self.num_cells = num_cells

    def _olustur_dengesiz_hucreler(self) -> List[BatteryCell]:
        cells = []
        # Hücreler arasında %75 ile %85 arasında dağılmış başlangıç SoC'si (~80 mV fark)
        soc_values = np.linspace(0.75, 0.85, self.num_cells)
        for i, s in enumerate(soc_values):
            c = BatteryCell(cell_id=i+1, voltage_v=0.0, capacity_ah=5.0, soc=s, temperature_c=25.0)
            c.update_voltage_from_soc()
            cells.append(c)
        return cells

    def benchmark_dengeleme(self) -> Dict[str, Any]:
        # 1. Pasif Dengeleme Simülasyonu
        passive_cells = self._olustur_dengesiz_hucreler()
        ctrl_passive = TeslaBalancingController(
            strategy=BalancingStrategy.PASSIVE_BLEEDING,
            voltage_threshold_mv=5.0,
            bleed_resistor_ohm=33.0
        )

        passive_imbalance_history = []
        passive_heat_history = []
        gecikmeler_step_us: List[float] = []

        # 3600 saniye (1 saat) simülasyon
        for s in range(3600):
            t0 = time.perf_counter_ns()
            res = ctrl_passive.step_balancing(passive_cells, dt_s=1.0)
            t1 = time.perf_counter_ns()
            gecikmeler_step_us.append(float(t1 - t0) / 1000.0)

            passive_imbalance_history.append(res["imbalance_mv"])
            passive_heat_history.append(res["heat_w"])
            if res["imbalance_mv"] <= 5.0:
                break

        # 2. Aktif Dengeleme Simülasyonu
        active_cells = self._olustur_dengesiz_hucreler()
        ctrl_active = TeslaBalancingController(
            strategy=BalancingStrategy.ACTIVE_INDUCTIVE,
            voltage_threshold_mv=5.0,
            active_transfer_current_a=2.0,
            active_efficiency=0.88
        )

        active_imbalance_history = []
        active_heat_history = []

        for s in range(3600):
            res = ctrl_active.step_balancing(active_cells, dt_s=1.0)
            active_imbalance_history.append(res["imbalance_mv"])
            active_heat_history.append(res["heat_w"])
            if res["imbalance_mv"] <= 5.0:
                break

        step_dizi = np.array(gecikmeler_step_us)
        t_step_avg_us = float(np.mean(step_dizi))

        return {
            "dengeleme_step_ortalama_us": t_step_avg_us,
            "dengeleme_step_p99_us": float(np.percentile(step_dizi, 99)),
            "saniyelik_dengeleme_adimi": int(1e6 / max(t_step_avg_us, 1e-4)),
            "passive_duration_s": len(passive_imbalance_history),
            "active_duration_s": len(active_imbalance_history),
            "passive_total_heat_j": float(np.sum(passive_heat_history)),
            "active_total_heat_j": float(np.sum(active_heat_history)),
            "passive_imbalance": passive_imbalance_history,
            "active_imbalance": active_imbalance_history,
            "speedup_factor": len(passive_imbalance_history) / max(len(active_imbalance_history), 1),
            "heat_saving_factor": float(np.sum(passive_heat_history)) / max(float(np.sum(active_heat_history)), 1e-4)
        }
