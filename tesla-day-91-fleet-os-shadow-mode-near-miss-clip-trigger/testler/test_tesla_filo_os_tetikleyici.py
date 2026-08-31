"""
Tesla Filo OS Tetikleyici Birim Testleri (PyTest)
=================================================
Bu test paketi; sert frenleme, acil direksiyon kaçışı ve gölge mod
insan-FSD sapma tetikleyicilerini test eder.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_filo_os_tetikleyici import TeslaFleetOSClipTrigger, FleetTelemetryEvent


def test_sert_fren_tetikleyicisi():
    """0.95g sert frenlemede 15 saniyelik klip paketlendiği test edilir."""
    trigger = TeslaFleetOSClipTrigger(g_force_thresh=0.8)
    evt = FleetTelemetryEvent(
        vin="5YJ3E1EB000001",
        timestamp_s=1700000050.0,
        g_force_decel=0.95,
        steering_rate_deg_s=10.0,
        human_accel_m_s2=-8.0,
        fsd_accel_m_s2=-8.0
    )

    trig, reason = trigger.evaluate_telemetry_event(evt)
    assert trig is True
    assert "HARD_BRAKING_EVENT" in reason

    pkg = trigger.package_15s_clip(evt.vin, reason, evt.timestamp_s)
    assert pkg["duration_s"] == 15.0
    assert pkg["clip_start_timestamp"] == 1700000040.0
    assert pkg["clip_end_timestamp"] == 1700000055.0


def test_acil_direksiyon_ve_golge_mod_tetikleyicisi():
    """Ani direksiyon ve insan-FSD sapmasının başarıyla yakalandığı test edilir."""
    trigger = TeslaFleetOSClipTrigger()

    # 1. Acil Direksiyon (240°/s)
    evt_steer = FleetTelemetryEvent(
        vin="5YJ3E1EB000002", timestamp_s=100.0, g_force_decel=0.2,
        steering_rate_deg_s=-240.0, human_accel_m_s2=0.0, fsd_accel_m_s2=0.0
    )
    trig1, reason1 = trigger.evaluate_telemetry_event(evt_steer)
    assert trig1 is True
    assert "EMERGENCY_STEERING" in reason1

    # 2. Gölge Mod Sapması (İnsan frene bastı, FSD gaza bastı -> Fark = 4.0 m/s²)
    evt_shadow = FleetTelemetryEvent(
        vin="5YJ3E1EB000003", timestamp_s=200.0, g_force_decel=0.3,
        steering_rate_deg_s=15.0, human_accel_m_s2=-3.0, fsd_accel_m_s2=1.0
    )
    trig2, reason2 = trigger.evaluate_telemetry_event(evt_shadow)
    assert trig2 is True
    assert "SHADOW_MODE_DISCREPANCY" in reason2


def test_normal_surus_tetiklenmeme():
    """Normal sürüşte hiçbir klibin tetiklenmediği test edilir."""
    trigger = TeslaFleetOSClipTrigger()
    evt_norm = FleetTelemetryEvent(
        vin="5YJ3E1EB000004", timestamp_s=300.0, g_force_decel=0.15,
        steering_rate_deg_s=12.0, human_accel_m_s2=0.5, fsd_accel_m_s2=0.5
    )
    trig, reason = trigger.evaluate_telemetry_event(evt_norm)
    assert trig is False
    assert reason == "NORMAL_CRUISE"
