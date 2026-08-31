"""
Tesla Phone Key UWB Birim Testleri (PyTest)
===========================================
Bu test paketi; UWB Time-of-Flight mesafe dönüşümünü,
BLE+UWB kilit açma yetkilendirmesini ve röle saldırısı tespitini test eder.

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

from src.tesla_phone_key_uwb_dogrulayici import TeslaPhoneKeyUWBValidator


def test_tof_mesafe_hesaplama():
    """4.5 ns ToF süresinin 1.35 metre mesafeye dönüştüğü test edilir."""
    validator = TeslaPhoneKeyUWBValidator()
    dist = validator.compute_distance_from_tof(4.5)

    assert np.isclose(dist, 1.35, atol=1e-3)
    assert validator.verify_uwb_distance(4.5) is True
    assert validator.verify_uwb_distance(10.0) is False  # 3.0m > 2.0m


def test_normal_yaklasim_kilit_acma():
    """Yetkili kullanıcının 1.35m yaklaştığında kilidin açıldığı test edilir."""
    validator = TeslaPhoneKeyUWBValidator()
    res = validator.evaluate_phone_key_unlock(ble_rssi_dbm=-60.0, uwb_tof_ns=4.5)

    assert res["door_unlocked"] is True
    assert res["relay_attack_detected"] is False
    assert res["ble_passed"] is True
    assert res["uwb_passed"] is True


def test_role_saldirisi_engelleme():
    """Yüksek RSSI fakat uzak ToF süresinde röle saldırısının yakalandığı ve kilidin açılmadığı test edilir."""
    validator = TeslaPhoneKeyUWBValidator()
    # RSSI çok güçlü (-45 dBm) fakat mesafe 35.0 ns = 10.5 metre
    res = validator.evaluate_phone_key_unlock(ble_rssi_dbm=-45.0, uwb_tof_ns=35.0)

    assert res["door_unlocked"] is False
    assert res["relay_attack_detected"] is True
    assert res["ble_passed"] is True
    assert res["uwb_passed"] is False
