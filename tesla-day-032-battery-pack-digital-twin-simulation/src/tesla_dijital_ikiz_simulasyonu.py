"""
Tesla Batarya Paketi Dijital İkiz (Digital Twin) Simülasyonu
============================================================
Bu modül; 96S hücre serisinden oluşan 400V batarya paketinin fiziksel
üretim varyasyonlarını, soğutma plakası termal gradyanını ve tekil hücre
termal kaçak (Thermal Runaway) anomali erken uyarısını gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class TwinCell:
    cell_id: int
    nominal_capacity_ah: float
    r0_ohm: float
    soc: float
    v_terminal: float
    temperature_c: float
    is_faulty: bool = False

    def update_step(self, current_a: float, dt_s: float, ambient_temp_c: float):
        # SoC güncelleme
        self.soc -= (current_a * dt_s) / (self.nominal_capacity_ah * 3600.0)
        self.soc = float(np.clip(self.soc, 0.001, 0.999))

        # OCV
        ocv = 3.0 + 1.20 * self.soc + 0.05 * np.log(self.soc) - 0.02 * np.exp(-15.0 * self.soc)
        # Voltaj
        r_actual = self.r0_ohm if not self.is_faulty else self.r0_ohm * 4.0
        self.v_terminal = float(ocv - current_a * r_actual)

        # Isı üretimi: Joule ısısı
        p_heat = (current_a ** 2) * r_actual
        c_th = 450.0  # Tek hücre ısıl kapasitesi J/K
        # Soğutma etkisi
        q_cool = (self.temperature_c - ambient_temp_c) * 1.5
        d_temp = ((p_heat - q_cool) * dt_s) / c_th
        self.temperature_c += float(d_temp)


class TeslaBatteryPackDigitalTwin:
    """
    96S Batarya Paketi Dijital İkiz Motoru.
    """
    def __init__(self, num_series_cells: int = 96, seed: int = 42):
        np.random.seed(seed)
        self.cells: List[TwinCell] = []

        # Gauss Üretim Varyasyonları: Q ~ N(75, 1.0), R0 ~ N(1.5mΩ, 0.1mΩ)
        for i in range(num_series_cells):
            q_nom = float(np.random.normal(75.0, 1.0))
            r0 = float(np.random.normal(0.0015, 0.0001))
            # Termal gradyan: Giriş (1. hücre) soğuk 22°C, Çıkış (96. hücre) 27°C
            t_init = 22.0 + (i / num_series_cells) * 5.0

            c = TwinCell(
                cell_id=i + 1,
                nominal_capacity_ah=q_nom,
                r0_ohm=r0,
                soc=0.85,
                v_terminal=4.0,
                temperature_c=t_init
            )
            self.cells.append(c)

    def inject_thermal_anomaly(self, cell_id: int = 48):
        """Hücrede mikro-iç kısa devre ve yüksek iç direnç anomalisi enjekte eder."""
        if 1 <= cell_id <= len(self.cells):
            self.cells[cell_id - 1].is_faulty = True

    def step(self, pack_current_a: float, dt_s: float = 0.1) -> Dict[str, Any]:
        """Tüm 96S hücrelerini simüle eder ve paket telemetrisini üretir."""
        voltages = []
        temps = []

        for i, cell in enumerate(self.cells):
            # Soğutma sıvısı gradyanı
            coolant_t = 20.0 + (i / len(self.cells)) * 4.0
            cell.update_step(pack_current_a, dt_s, coolant_t)
            voltages.append(cell.v_terminal)
            temps.append(cell.temperature_c)

        v_pack = float(np.sum(voltages))
        v_min = float(np.min(voltages))
        v_max = float(np.max(voltages))
        t_max = float(np.max(temps))
        t_min = float(np.min(temps))
        imbalance_mv = (v_max - v_min) * 1000.0

        # Erken Uyarı Anomali Tespiti:
        # Herhangi bir hücrenin sıcaklığı ortalamadan 8°C fazlaysa veya voltajı 150 mV çökmüşse
        t_mean = np.mean(temps)
        anomaly_detected = False
        faulty_cell_id = None

        for cell in self.cells:
            if (cell.temperature_c - t_mean) > 3.0 or (v_max - cell.v_terminal) > 0.060:
                anomaly_detected = True
                faulty_cell_id = cell.cell_id
                break

        return {
            "v_pack": v_pack,
            "v_min": v_min,
            "v_max": v_max,
            "imbalance_mv": imbalance_mv,
            "t_max": t_max,
            "t_min": t_min,
            "t_gradient_c": t_max - t_min,
            "anomaly_flag": anomaly_detected,
            "faulty_cell_id": faulty_cell_id,
            "cell_voltages": voltages,
            "cell_temperatures": temps
        }
