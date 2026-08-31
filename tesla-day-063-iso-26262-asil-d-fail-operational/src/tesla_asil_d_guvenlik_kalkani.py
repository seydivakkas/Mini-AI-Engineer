r"""
Tesla ISO 26262 ASIL-D Fonksiyonel Güvenlik ve Fail-Operational Kalkanı
========================================================================
Bu modül; Çift Kanallı Sensör Çapraz Doğrulamasını (Dual-Channel Cross-Check),
Arıza Filtreleme (Fault Debouncing), ASIL-D Güvenlik Bayrağı Üretimini ve
Arıza Durumunda Minimal Risk Manevrasını (Fail-Operational MRM Safe Stop) gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np


class SafetyState(Enum):
    NOMINAL = "NOMINAL"
    DEGRADED_WARNING = "DEGRADED_WARNING"
    ASIL_D_FAULT_TRIGGERED = "ASIL_D_FAULT_TRIGGERED"
    FAIL_OPERATIONAL_SAFE_STOP = "FAIL_OPERATIONAL_SAFE_STOP"


class TeslaASILDSafetyGuard:
    """
    Tesla FSD ISO 26262 ASIL-D Fonksiyonel Güvenlik Kalkanı.
    """
    def __init__(
        self,
        max_torque_diff_nm: float = 0.50,    # Çift kanal tork uyuşmazlık limiti (0.5 Nm)
        max_speed_diff_mps: float = 0.40,    # Çift kanal hız uyuşmazlık limiti (0.4 m/s)
        fault_debounce_threshold: int = 3    # 3 ardışık çevrim arıza şartı
    ):
        self.max_torque_diff = max_torque_diff_nm
        self.max_speed_diff = max_speed_diff_mps
        self.debounce_limit = fault_debounce_threshold
        self.consecutive_fault_count = 0
        self.current_state = SafetyState.NOMINAL

    def check_dual_channel_asil_d(self, val_ch1: float, val_ch2: float, max_diff: float = 0.50) -> bool:
        """
        Çift Kanallı Donanım Sinyali Çapraz Doğrulaması:
        |S1 - S2| <= max_diff
        """
        return bool(abs(val_ch1 - val_ch2) <= max_diff)

    def process_safety_cycle(
        self,
        torque_ch1_nm: float,
        torque_ch2_nm: float,
        speed_ch1_mps: float,
        speed_ch2_mps: float
    ) -> Dict[str, Any]:
        """
        Her 10ms RTOS döngüsünde ASIL-D çift kanal doğrulaması ve arıza yönetimi.
        """
        torque_ok = self.check_dual_channel_asil_d(torque_ch1_nm, torque_ch2_nm, self.max_torque_diff)
        speed_ok = self.check_dual_channel_asil_d(speed_ch1_mps, speed_ch2_mps, self.max_speed_diff)

        is_cycle_fault = not (torque_ok and speed_ok)

        if is_cycle_fault:
            self.consecutive_fault_count += 1
            if self.consecutive_fault_count >= self.debounce_limit:
                self.current_state = SafetyState.FAIL_OPERATIONAL_SAFE_STOP
            else:
                self.current_state = SafetyState.DEGRADED_WARNING
        else:
            self.consecutive_fault_count = max(0, self.consecutive_fault_count - 1)
            if self.consecutive_fault_count == 0:
                self.current_state = SafetyState.NOMINAL

        # Fail-Operational Aksiyon Belirleme
        if self.current_state == SafetyState.FAIL_OPERATIONAL_SAFE_STOP:
            mrm_action = "GÜVENLİ DURUŞ MANEVRASI (MRM -1.5 m/s², Flaşörler Açık, Emniyet Şeridi)"
            is_drive_allowed = False
        elif self.current_state == SafetyState.DEGRADED_WARNING:
            mrm_action = "UYARI: Kanal Uyuşmazlığı Filtreleniyor"
            is_drive_allowed = True
        else:
            mrm_action = "NOMİNAL: Tüm Güvenlik Kanalları Sağlam"
            is_drive_allowed = True

        return {
            "safety_state": self.current_state.value,
            "torque_ch1_nm": torque_ch1_nm,
            "torque_ch2_nm": torque_ch2_nm,
            "torque_diff_nm": float(abs(torque_ch1_nm - torque_ch2_nm)),
            "speed_diff_mps": float(abs(speed_ch1_mps - speed_ch2_mps)),
            "fault_count": self.consecutive_fault_count,
            "mrm_action": mrm_action,
            "is_drive_allowed": is_drive_allowed,
            "is_safe": bool(self.current_state == SafetyState.NOMINAL)
        }
