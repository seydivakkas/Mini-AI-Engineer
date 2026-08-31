r"""
Tesla Gölge Modu (Shadow Mode) ve Veri Motoru (Data Engine) Çekirdeği
======================================================================
Bu modül; Araç içinde sessizce çalışan Gölge Model ile İnsan Sürücü
eylemleri arasındaki uyuşmazlık tetikleyicilerini (Discrepancy Triggers),
Uç Klip Paketleme (Edge Snapshot Buffering) ve A/B İstatistik Testlerini gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaShadowModeDataEngine:
    """
    Tesla FSD Gölge Modu ve Uç Tetikleme Motoru.
    """
    def __init__(
        self,
        steering_thresh_deg: float = 5.0,
        accel_thresh_mps2: float = 1.5
    ):
        self.steering_thresh = steering_thresh_deg
        self.accel_thresh = accel_thresh_mps2

    def check_discrepancy_and_trigger(
        self,
        human_steering_deg: float,
        shadow_steering_deg: float,
        human_accel_mps2: float,
        shadow_accel_mps2: float,
        human_lane_action: str = "KEEP",
        shadow_lane_action: str = "KEEP"
    ) -> Dict[str, Any]:
        """
        İnsan sürücü ile gölge model arasındaki karar farkını denetler.
        """
        steer_diff = abs(human_steering_deg - shadow_steering_deg)
        accel_diff = abs(human_accel_mps2 - shadow_accel_mps2)
        lane_diff = (human_lane_action != shadow_lane_action)

        steer_trigger = steer_diff > self.steering_thresh
        accel_trigger = accel_diff > self.accel_thresh
        lane_trigger = lane_diff

        is_triggered = bool(steer_trigger or accel_trigger or lane_trigger)

        trigger_reasons = []
        if steer_trigger:
            trigger_reasons.append(f"STEERING_DELTA_{steer_diff:.1f}deg")
        if accel_trigger:
            trigger_reasons.append(f"ACCEL_DELTA_{accel_diff:.1f}mps2")
        if lane_trigger:
            trigger_reasons.append(f"LANE_MISMATCH_{human_lane_action}_vs_{shadow_lane_action}")

        # Tetikleme anında veri motoruna yüklenecek klip paketi
        clip_package = None
        if is_triggered:
            clip_package = {
                "timestamp_ms": 1725100000000,
                "buffer_window": "[-10s, +5s]",
                "camera_views": 8,
                "imu_can_telemetry": True,
                "reasons": trigger_reasons,
                "payload_size_mb": 42.5
            }

        return {
            "is_triggered": is_triggered,
            "steering_diff_deg": steer_diff,
            "accel_diff_mps2": accel_diff,
            "lane_action_diff": lane_diff,
            "trigger_reasons": trigger_reasons,
            "clip_package": clip_package
        }

    def evaluate_ab_test_significance(
        self,
        interventions_model_a: int,
        miles_model_a: float,
        interventions_model_b: int,
        miles_model_b: float
    ) -> Dict[str, Any]:
        """
        İki model arasındaki Müdahale Başına Mil (Miles Per Intervention - MPI) Z-Testi.
        """
        rate_a = interventions_model_a / max(miles_model_a, 1.0)
        rate_b = interventions_model_b / max(miles_model_b, 1.0)

        # Havuzlanmış oran
        p_pool = (interventions_model_a + interventions_model_b) / (miles_model_a + miles_model_b)
        se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0/miles_model_a + 1.0/miles_model_b))

        z_score = (rate_a - rate_b) / max(se, 1e-7)
        import math
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / np.sqrt(2.0))))

        mpi_a = 1.0 / max(rate_a, 1e-6)
        mpi_b = 1.0 / max(rate_b, 1e-6)

        return {
            "mpi_model_a": mpi_a,
            "mpi_model_b": mpi_b,
            "improvement_pct": ((mpi_b - mpi_a) / mpi_a) * 100.0,
            "z_score": float(z_score),
            "p_value": float(p_val),
            "statistically_significant": bool(p_val < 0.05)
        }
