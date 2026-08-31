"""
Tesla D-Bus BCM Birim Testleri (PyTest)
=======================================
Bu test paketi; D-Bus RPC metod çağrılarını, asenkron sinyal yayılımını
ve kapı/pencere gövde kontrol parametrelerini test eder.

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

from src.tesla_dbus_bcm_yonetici import TeslaDBusBodyController, LightMode


def test_kapi_kilidi_rpc_ve_sinyal():
    """Kapı kilidi değiştirildiğinde D-Bus sinyalinin yayınlandığı test edilir."""
    bcm = TeslaDBusBodyController()
    basarili = bcm.set_door_lock("FRONT_LEFT", False)

    assert basarili is True
    assert bcm.doors["FRONT_LEFT"] is False
    assert len(bcm.dbus_signal_log) == 1
    assert bcm.dbus_signal_log[0]["signal"] == "DoorStatusChanged"
    assert bcm.dbus_signal_log[0]["params"]["locked"] is False


def test_cam_pozisyonu_sinirlandirma():
    """Cam pozisyonunun %0 - %100 arasına sınırlandığı test edilir."""
    bcm = TeslaDBusBodyController()
    bcm.set_window_position("FRONT_RIGHT", 145.0)  # Aşırı değer

    assert bcm.windows["FRONT_RIGHT"] == 100.0
    assert bcm.dbus_signal_log[-1]["params"]["position_pct"] == 100.0


def test_far_ve_sarj_portu_metodlari():
    """Far modu ve şarj portu metodlarının çalıştığı test edilir."""
    bcm = TeslaDBusBodyController()
    bcm.set_lights_mode(LightMode.HIGH_BEAM)
    bcm.set_charge_port(True)

    assert bcm.lights_mode == LightMode.HIGH_BEAM
    assert bcm.charge_port_open is True
