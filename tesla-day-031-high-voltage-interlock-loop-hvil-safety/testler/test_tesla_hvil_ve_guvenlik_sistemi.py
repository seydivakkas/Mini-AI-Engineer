"""
Tesla HVIL ve Güvenlik Sistemi Birim Testleri (PyTest)
======================================================
Bu test paketi; HVIL döngü arıza tespiti, Precharge sıralaması, Pyrofuse
kaza patlatması ve izolasyon kaçağı korumasını test eder.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import numpy as np
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_hvil_ve_guvenlik_sistemi import (
    TeslaHVILSafetyManager,
    HighVoltageSystemState,
    ContactorState,
    HVILStatus
)


def test_hvil_sinyal_degerlendirme():
    """88 Hz PWM sinyalinin sağlıklı, 0V'un açık devre olarak algılandığı test edilir."""
    mgr = TeslaHVILSafetyManager()

    # Sağlıklı 88 Hz
    assert mgr.evaluate_hvil_loop(pwm_freq_hz=88.0, duty_pct=50.0, loop_voltage_v=5.0) == HVILStatus.LOOP_CLOSED_HEALTHY
    # Açık Devre (0V)
    assert mgr.evaluate_hvil_loop(pwm_freq_hz=0.0, duty_pct=0.0, loop_voltage_v=0.0) == HVILStatus.LOOP_OPEN_FAULT
    # 12V Kısa Devre
    assert mgr.evaluate_hvil_loop(pwm_freq_hz=0.0, duty_pct=100.0, loop_voltage_v=12.0) == HVILStatus.SHORT_TO_12V


def test_precharge_ve_ana_kontaktor_siralamasi():
    """Precharge tamamlandığında ana kontaktörlerin güvenle kapatıldığı test edilir."""
    mgr = TeslaHVILSafetyManager()
    state = HighVoltageSystemState(v_battery_dc=400.0, v_inverter_link=0.0)

    # 300 ms boyunca adım işlet
    for _ in range(300):
        out = mgr.execute_safety_cycle(state, dt_ms=1.0)

    assert out["safe"] is True
    assert state.contactor_state == ContactorState.MAIN_CLOSED_ENERGIZED
    assert state.v_inverter_link >= 380.0


def test_kaza_aninda_pyrofuse_patlatma():
    """RCM kaza sinyali geldiğinde Pyrofuse'un patlatılıp gücün anında kesildiği test edilir."""
    mgr = TeslaHVILSafetyManager()
    state = HighVoltageSystemState(v_battery_dc=400.0, crash_signal_rcm=True)

    out = mgr.execute_safety_cycle(state, dt_ms=1.0)
    assert out["safe"] is False
    assert state.pyrofuse_intact is False
    assert state.contactor_state == ContactorState.PYROFUSE_BLOWN
    assert out["fault"] == "CRASH_PYROFUSE_TRIGGERED"


def test_izolasyon_kaybinda_guc_kesme():
    """İzolasyon direnci 200 kOhm altına indiğinde kontaktörlerin açıldığı test edilir."""
    mgr = TeslaHVILSafetyManager(min_isolation_kohm=200.0)
    state = HighVoltageSystemState(v_battery_dc=400.0, r_isolation_kohm=50.0)  # Kaçak var

    out = mgr.execute_safety_cycle(state, dt_ms=1.0)
    assert out["safe"] is False
    assert state.contactor_state == ContactorState.ALL_OPEN
    assert out["fault"] == "ISOLATION_LOSS_CHASSIS_LEAKAGE"
