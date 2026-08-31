"""
Tesla Gölge Profilleyici Modülü
===============================
Bu modül; Shadow Mode uyuşmazlık tetikleme hızını, veri paketi üretimini
ve A/B hipotez testi hesaplama süresini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_golge_modu_ve_veri_motoru import TeslaShadowModeDataEngine


class TeslaGolgeProfilleyici:
    """
    Shadow Mode ve Veri Motoru Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_shadow_mode(self) -> Dict[str, Any]:
        engine = TeslaShadowModeDataEngine(steering_thresh_deg=5.0, accel_thresh_mps2=1.5)

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            # Örnek uyuşmazlık: İnsan sola kaçıyor (6.5 deg), Gölge düz kalıyor
            ciktilar = engine.check_discrepancy_and_trigger(
                human_steering_deg=-6.5,
                shadow_steering_deg=0.0,
                human_accel_mps2=-1.8,
                shadow_accel_mps2=-0.1,
                human_lane_action="CHANGE_LEFT",
                shadow_lane_action="KEEP"
            )
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        # A/B Testi İstatistiği (Model A: v11.4 vs Model B: v12.3)
        ab_res = engine.evaluate_ab_test_significance(
            interventions_model_a=50, miles_model_a=10000.0,
            interventions_model_b=15, miles_model_b=10000.0
        )

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "shadow_step_ortalama_us": t_avg_us,
            "shadow_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_denetim_adimi": int(1e6 / max(t_avg_us, 1e-4)),
            "is_triggered": ciktilar["is_triggered"],
            "steer_diff": ciktilar["steering_diff_deg"],
            "accel_diff": ciktilar["accel_diff_mps2"],
            "trigger_reasons": ciktilar["trigger_reasons"],
            "ab_test": ab_res,
            "gecikmeler": gecikmeler_us[:200]
        }
