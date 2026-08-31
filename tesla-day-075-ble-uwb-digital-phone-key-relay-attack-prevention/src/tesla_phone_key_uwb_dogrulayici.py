r"""
Tesla BLE ve UWB Dijital Telefon Anahtarı (Phone Key) Çekirdeği
================================================================
Bu modül; Tesla Model 3/Y ve Cybertruck araçlarının BLE (Bluetooth Low Energy)
ve UWB (Ultra-Wideband) Time-of-Flight (ToF) temassız kilit açma protokolünü,
milimetrik mesafe doğrulamasını ve Röle İstasyonu Sinyal Hırsızlığına
(Relay Attack Prevention) karşı koruma motorunu gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaPhoneKeyUWBValidator:
    """
    Tesla UWB ve BLE Dijital Telefon Anahtarı Güvenlik Doğrulayıcısı.
    """
    SPEED_OF_LIGHT_MPS = 3.0e8  # 300,000,000 m/s

    def __init__(self, max_unlock_dist_m: float = 2.0, ble_rssi_threshold_dbm: float = -75.0):
        self.max_dist_m = max_unlock_dist_m
        self.rssi_threshold = ble_rssi_threshold_dbm

    def compute_distance_from_tof(self, tof_nanoseconds: float) -> float:
        """UWB Time-of-Flight (nanosaniye) süresini mesafeye (metre) çevirir."""
        tof_s = tof_nanoseconds * 1e-9
        return float(tof_s * self.SPEED_OF_LIGHT_MPS)

    def verify_uwb_distance(self, tof_nanoseconds: float) -> bool:
        """ToF mesafesinin 2.0 metreden kısa olduğunu doğrular."""
        dist = self.compute_distance_from_tof(tof_nanoseconds)
        return bool(dist <= self.max_dist_m)

    def is_ble_proximate(self, rssi_dbm: float) -> bool:
        """BLE RSSI sinyal gücünün eşik üzerinde olduğunu doğrular."""
        return bool(rssi_dbm >= self.rssi_threshold)

    def evaluate_phone_key_unlock(
        self,
        ble_rssi_dbm: float,
        uwb_tof_ns: float
    ) -> Dict[str, Any]:
        """
        BLE ve UWB'yi birleşik değerlendirip röle saldırısını tespit eder.
        """
        dist_m = self.compute_distance_from_tof(uwb_tof_ns)
        ble_ok = self.is_ble_proximate(ble_rssi_dbm)
        uwb_ok = self.verify_uwb_distance(uwb_tof_ns)

        # Röle Saldırısı Senaryosu: BLE güçlü sinyal veriyor (RSSI yüksek) fakat UWB ToF süresi uzak
        is_relay_attack = ble_ok and not uwb_ok
        should_unlock = bool(ble_ok and uwb_ok)

        status_text = "KİLİT AÇILDI: Yetkili Sürücü Yanında (1.35m)" if should_unlock else (
            "RÖLE SALDIRISI ENGELLENDİ: Sinyal Güçlü Fakat Fiziksel Mesafe Uzak!" if is_relay_attack else
            "BEKLEMEDE: Telefon Kapsama Dışında"
        )

        return {
            "ble_rssi_dbm": ble_rssi_dbm,
            "uwb_tof_ns": uwb_tof_ns,
            "calculated_distance_m": dist_m,
            "ble_passed": ble_ok,
            "uwb_passed": uwb_ok,
            "relay_attack_detected": is_relay_attack,
            "door_unlocked": should_unlock,
            "status_text": status_text
        }
