"""
Tesla ISO 26262 ASIL-D Birim Testleri (PyTest)
==============================================
Bu test paketi; Çift kanal sensör doğrulamasını, debounce hata biriktirme
mantığını ve Fail-Operational MRM güvenli duruş geçişlerini test eder.

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

from src.tesla_asil_d_guvenlik_kalkani import TeslaASILDSafetyGuard, SafetyState


def test_cift_kanal_dogrulama():
    """Tork farkı 0.5 Nm altındayken True, üstündeyken False döndüğü test edilir."""
    guard = TeslaASILDSafetyGuard(max_torque_diff_nm=0.50)

    assert guard.check_dual_channel_asil_d(val_ch1=2.1, val_ch2=2.3, max_diff=0.50) is True
    assert guard.check_dual_channel_asil_d(val_ch1=2.1, val_ch2=2.9, max_diff=0.50) is False


def test_nominal_guvenlik_durumu():
    """Kanal sinyalleri uyumluyken NOMINAL durumda kalındığı ve sürüşe izin verildiği test edilir."""
    guard = TeslaASILDSafetyGuard()
    res = guard.process_safety_cycle(torque_ch1_nm=2.1, torque_ch2_nm=2.2, speed_ch1_mps=25.0, speed_ch2_mps=25.1)

    assert res["safety_state"] == SafetyState.NOMINAL.value
    assert res["is_safe"] is True
    assert res["is_drive_allowed"] is True


def test_ardisik_ariza_ve_fail_operational_tetikleme():
    """3 ardışık arıza çevrimi sonunda FAIL_OPERATIONAL_SAFE_STOP tetiklendiği test edilir."""
    guard = TeslaASILDSafetyGuard(fault_debounce_threshold=3)

    # 1. ve 2. Çevrim: Uyarı
    guard.process_safety_cycle(torque_ch1_nm=2.1, torque_ch2_nm=3.0, speed_ch1_mps=25.0, speed_ch2_mps=25.0)
    guard.process_safety_cycle(torque_ch1_nm=2.1, torque_ch2_nm=3.0, speed_ch1_mps=25.0, speed_ch2_mps=25.0)

    # 3. Çevrim: Kritik ASIL-D Arıza
    res3 = guard.process_safety_cycle(torque_ch1_nm=2.1, torque_ch2_nm=3.0, speed_ch1_mps=25.0, speed_ch2_mps=25.0)

    assert res3["safety_state"] == SafetyState.FAIL_OPERATIONAL_SAFE_STOP.value
    assert res3["is_drive_allowed"] is False
    assert res3["is_safe"] is False
