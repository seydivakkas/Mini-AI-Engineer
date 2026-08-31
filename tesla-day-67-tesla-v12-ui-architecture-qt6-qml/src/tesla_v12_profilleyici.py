"""
Tesla V12 UI Profilleyici Modülü
================================
Bu modül; C++ Backend Q_PROPERTY sinyal yayılım hızını,
QML veri aktarım süresini ve 60 FPS (16.6 ms) bütçe uyumluluğunu profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_v12_ui_model import TeslaV12VehicleModel


class TeslaV12UIProfilleyici:
    """
    Tesla V12 UI Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_ui_model(self) -> Dict[str, Any]:
        model = TeslaV12VehicleModel()

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = model.simulate_ui_stream(frames=60)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "ui_frame_ortalama_us": t_avg_us / 60.0,
            "ui_stream_ortalama_us": t_avg_us,
            "ui_stream_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_kare_isleme": int(1e6 / max(t_avg_us / 60.0, 1e-4)),
            "final_speed": ciktilar["final_speed_kmh"],
            "battery_pct": ciktilar["final_battery_pct"],
            "gear": ciktilar["final_gear"],
            "fsd_active": ciktilar["fsd_active"],
            "speeds": ciktilar["speeds_stream"],
            "signals": ciktilar["total_signals_emitted"],
            "is_60fps": ciktilar["is_60fps_ready"],
            "gecikmeler": gecikmeler_us[:200]
        }
