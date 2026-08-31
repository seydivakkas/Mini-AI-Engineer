"""
Tesla Filo Profilleyici Modülü
==============================
Bu modül; Filo OS telemetri tetikleyicilerinin çözümleme hızını ve
Map-Reduce filtreleme gecikmesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_filo_os_tetikleyici import TeslaFleetOSClipTrigger, FleetTelemetryEvent


class TeslaFiloProfilleyici:
    """
    Tesla Fleet OS Performans Profilleyicisi.
    """
    def __init__(self, fleet_event_count: int = 5000):
        self.count = fleet_event_count

    def generate_synthetic_fleet_events(self) -> List[FleetTelemetryEvent]:
        events = []
        for i in range(self.count):
            # %5 kritik sert fren, %3 acil direksiyon, %2 gölge sapma, %90 normal
            r = np.random.rand()
            if r < 0.05:
                g = float(np.random.uniform(0.82, 1.2))
                steer = float(np.random.uniform(20.0, 50.0))
                h_acc = -7.5
                f_acc = -7.4
            elif r < 0.08:
                g = 0.3
                steer = float(np.random.uniform(210.0, 320.0))
                h_acc = 0.0
                f_acc = 0.1
            elif r < 0.10:
                g = 0.4
                steer = 30.0
                h_acc = -3.5
                f_acc = 0.2  # İnsan frene bastı, FSD basmadı -> Sapma!
            else:
                g = float(np.random.uniform(0.05, 0.4))
                steer = float(np.random.uniform(5.0, 45.0))
                h_acc = 0.5
                f_acc = 0.5

            events.append(FleetTelemetryEvent(
                vin=f"5YJ3E1EB{i:06d}",
                timestamp_s=1700000000.0 + i,
                g_force_decel=g,
                steering_rate_deg_s=steer,
                human_accel_m_s2=h_acc,
                fsd_accel_m_s2=f_acc
            ))
        return events

    def benchmark_fleet_trigger(self) -> Dict[str, Any]:
        trigger = TeslaFleetOSClipTrigger()
        events = self.generate_synthetic_fleet_events()

        gecikmeler_us: List[float] = []

        for _ in range(50):
            t0 = time.perf_counter_ns()
            _ = trigger.map_reduce_fleet_filter(events[:1000])
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        critical_pkgs = trigger.map_reduce_fleet_filter(events)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))
        per_event_us = t_avg_us / 1000.0

        return {
            "total_fleet_events": self.count,
            "critical_clips_triggered": len(critical_pkgs),
            "trigger_rate_pct": float(np.round((len(critical_pkgs) / self.count) * 100.0, 2)),
            "per_event_ortalama_us": per_event_us,
            "filter_1000_batch_us": t_avg_us,
            "filter_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_olay_tarama": int(1e6 / max(per_event_us, 1e-4)),
            "reasons_sample": [p["trigger_reason"] for p in critical_pkgs[:10]],
            "gecikmeler": gecikmeler_us[:200]
        }
