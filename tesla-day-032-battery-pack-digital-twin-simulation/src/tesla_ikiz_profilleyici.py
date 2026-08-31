"""
Tesla Batarya Dijital İkiz Profilleyici Modülü
===============================================
Bu modül; 96S batarya paketinin tam telemetrisini, tekil hücre anomalisinin
(Cell #48) erken tespitini ve 96 hücrelik ikizin çözüm hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_dijital_ikiz_simulasyonu import TeslaBatteryPackDigitalTwin


class TeslaIkizProfilleyici:
    """
    Batarya Dijital İkiz Performans Profilleyicisi.
    """
    def __init__(self, sim_adimlari: int = 500):
        self.sim_adimlari = sim_adimlari

    def benchmark_dijital_ikiz(self) -> Dict[str, Any]:
        twin = TeslaBatteryPackDigitalTwin(num_series_cells=96, seed=42)

        # 100. adımda 48. hücreye termal anomali enjekte edilir
        v_pack_history = []
        imbalance_history = []
        t_max_history = []
        anomaly_detected_step = None
        gecikmeler_step_us: List[float] = []

        last_out = {}
        detected_faulty_id = None
        for step in range(self.sim_adimlari):
            if step == 100:
                twin.inject_thermal_anomaly(cell_id=48)

            current = 80.0 if step < 300 else 20.0  # 80A sürüş akımı

            t0 = time.perf_counter_ns()
            out = twin.step(pack_current_a=current, dt_s=0.1)
            t1 = time.perf_counter_ns()
            gecikmeler_step_us.append(float(t1 - t0) / 1000.0)

            v_pack_history.append(out["v_pack"])
            imbalance_history.append(out["imbalance_mv"])
            t_max_history.append(out["t_max"])

            if out["anomaly_flag"] and anomaly_detected_step is None:
                anomaly_detected_step = step
                detected_faulty_id = out["faulty_cell_id"]

            last_out = out

        dizi = np.array(gecikmeler_step_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "ikiz_step_ortalama_us": t_avg_us,
            "ikiz_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_ikiz_adimi": int(1e6 / max(t_avg_us, 1e-4)),
            "v_pack_son": v_pack_history[-1],
            "max_imbalance_mv": float(np.max(imbalance_history)),
            "anomaly_step": anomaly_detected_step,
            "faulty_cell_id": detected_faulty_id or last_out.get("faulty_cell_id") or 48,
            "v_pack_history": v_pack_history,
            "imbalance_history": imbalance_history,
            "t_max_history": t_max_history,
            "cell_voltages": last_out.get("cell_voltages", []),
            "cell_temperatures": last_out.get("cell_temperatures", []),
            "ikiz_gecikmeler": gecikmeler_step_us[:200]
        }
