"""
Tesla Yüksek Gerilim Kilidi (HVIL), İzolasyon ve Güvenlik Sistemi
==================================================================
Bu modül; 400V/800V yüksek gerilim hattı için HVIL (High Voltage Interlock Loop)
döngüsü sürekliliğini, Pyrofuse acil durum patlatma kontrolünü, ISO 6469-1
izolasyon direnci takibini ve kontaktör ön şarj (Precharge) sıralamasını gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class HVILStatus(Enum):
    LOOP_CLOSED_HEALTHY = 1  # 88 Hz PWM sağlam ve kapalı döngü
    LOOP_OPEN_FAULT = 2      # Konnektör gevşek veya kapak açık (Açık Devre)
    SHORT_TO_GND = 3         # Şasiye kısa devre
    SHORT_TO_12V = 4         # 12V beslemeye kısa devre


class ContactorState(Enum):
    ALL_OPEN = "ALL_OPEN"
    PRECHARGE_ACTIVE = "PRECHARGE_ACTIVE"
    MAIN_CLOSED_ENERGIZED = "MAIN_CLOSED_ENERGIZED"
    WELDED_FAULT = "WELDED_FAULT"
    PYROFUSE_BLOWN = "PYROFUSE_BLOWN"


@dataclass
class HighVoltageSystemState:
    v_battery_dc: float = 400.0
    v_inverter_link: float = 0.0
    r_isolation_kohm: float = 500.0  # 500 kOhm (Sağlıklı)
    hvil_signal_valid: bool = True
    crash_signal_rcm: bool = False
    contactor_state: ContactorState = ContactorState.ALL_OPEN
    pyrofuse_intact: bool = True


class TeslaHVILSafetyManager:
    """
    Tesla ASIL-D Yüksek Gerilim Güvenlik Yöneticisi.
    """
    def __init__(self, min_isolation_kohm: float = 200.0, precharge_threshold_pct: float = 0.95):
        self.min_r_iso = min_isolation_kohm
        self.precharge_thresh = precharge_threshold_pct
        self.precharge_timer_ms = 0

    def evaluate_hvil_loop(self, pwm_freq_hz: float, duty_pct: float, loop_voltage_v: float) -> HVILStatus:
        """HVIL 88 Hz PWM sinyal denetimi."""
        if loop_voltage_v < 0.5:
            return HVILStatus.LOOP_OPEN_FAULT
        elif loop_voltage_v > 11.5 and pwm_freq_hz < 10.0:
            return HVILStatus.SHORT_TO_12V
        elif abs(pwm_freq_hz - 88.0) < 5.0 and 40.0 <= duty_pct <= 60.0:
            return HVILStatus.LOOP_CLOSED_HEALTHY
        else:
            return HVILStatus.SHORT_TO_GND

    def execute_safety_cycle(self, state: HighVoltageSystemState, dt_ms: float = 1.0) -> Dict[str, Any]:
        """1 kHz Yüksek Gerilim Güvenlik Döngüsü (1 ms)."""
        fault_detected = False
        fault_reason = "NONE"

        # 1. Kaza / Çarpışma Tespiti -> Pyrofuse Anında Ateşleme (< 5 ms)
        if state.crash_signal_rcm and state.pyrofuse_intact:
            state.pyrofuse_intact = False
            state.contactor_state = ContactorState.PYROFUSE_BLOWN
            state.v_inverter_link = 0.0
            return {
                "safe": False,
                "contactor_state": state.contactor_state.value,
                "fault": "CRASH_PYROFUSE_TRIGGERED",
                "hvil_ok": False,
                "isolation_ok": False,
                "v_link": 0.0
            }

        # 2. HVIL Döngü Güvenliği Denetimi
        if not state.hvil_signal_valid or not state.pyrofuse_intact:
            state.contactor_state = ContactorState.ALL_OPEN
            state.v_inverter_link = 0.0
            return {
                "safe": False,
                "contactor_state": state.contactor_state.value,
                "fault": "HVIL_INTERRUPTED_EMERGENCY_SHUTDOWN",
                "hvil_ok": False,
                "isolation_ok": state.r_isolation_kohm >= self.min_r_iso,
                "v_link": 0.0
            }

        # 3. İzolasyon Direnci Denetimi (ISO 6469-1: > 500 Ohm/V)
        if state.r_isolation_kohm < self.min_r_iso:
            state.contactor_state = ContactorState.ALL_OPEN
            return {
                "safe": False,
                "contactor_state": state.contactor_state.value,
                "fault": "ISOLATION_LOSS_CHASSIS_LEAKAGE",
                "hvil_ok": True,
                "isolation_ok": False,
                "v_link": 0.0
            }

        # 4. Ön Şarj (Precharge) ve Ana Kontaktör Kapatma Sıralaması
        if state.contactor_state == ContactorState.ALL_OPEN:
            state.contactor_state = ContactorState.PRECHARGE_ACTIVE
            self.precharge_timer_ms = 0

        if state.contactor_state == ContactorState.PRECHARGE_ACTIVE:
            self.precharge_timer_ms += dt_ms
            # Precharge direnci üzerinden invertör DC link kapasitörü şarj olur:
            tau_precharge_ms = 80.0
            v_target = state.v_battery_dc * (1.0 - np.exp(-self.precharge_timer_ms / tau_precharge_ms))
            state.v_inverter_link = float(v_target)

            # İnvertör gerilimi bataryanın %95'ine ulaştığında ana kontaktör kapatılır
            if state.v_inverter_link >= (state.v_battery_dc * self.precharge_thresh):
                state.contactor_state = ContactorState.MAIN_CLOSED_ENERGIZED

        return {
            "safe": True,
            "contactor_state": state.contactor_state.value,
            "fault": "NONE",
            "hvil_ok": True,
            "isolation_ok": True,
            "v_link": float(state.v_inverter_link),
            "precharge_time_ms": float(self.precharge_timer_ms)
        }
