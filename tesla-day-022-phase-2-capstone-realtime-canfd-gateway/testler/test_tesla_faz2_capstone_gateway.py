"""
Tesla Faz 2 Capstone Gateway Birim Testleri (PyTest)
====================================================
Bu test paketi; CAN-FD, LIN, SOME/IP ve UDS protokollerinin birleşik Gateway
üzerindeki yönlendirme ve telemetri doğruluklarını test eder.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import struct
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_faz2_capstone_gateway import TeslaCentralGateway, GatewayVehicleState


def test_canfd_powertrain_cozumu_ve_guc():
    """CAN-FD 0x301 üzerinden voltaj, akım ve güç hesabının doğruluğu test edilir."""
    gw = TeslaCentralGateway()
    # 400.0 V (4000), 100.0 A (1000), 32.5 C (325)
    payload = struct.pack(">HHh2x", 4000, 1000, 325)
    gw.decode_canfd_powertrain(0x301, payload)

    assert pytest.approx(gw.state.pack_voltage_v, 0.1) == 400.0
    assert pytest.approx(gw.state.pack_current_a, 0.1) == 100.0
    assert pytest.approx(gw.state.inverter_temp_c, 0.1) == 32.5
    # P = 400V * 100A / 1000 = 40.0 kW
    assert pytest.approx(gw.state.power_kw, 0.1) == 40.0


def test_canfd_chassis_ve_hiz_cozumu():
    """CAN-FD 0x12F üzerinden hız ve direksiyon açısı çözümü test edilir."""
    gw = TeslaCentralGateway()
    # Hız: 120.0 km/h (2400 * 0.05), Açı: 0.0 deg (1800 * 0.1 - 180.0)
    payload = struct.pack(">HH4x", 2400, 1800)
    gw.decode_canfd_chassis(0x12F, payload)

    assert pytest.approx(gw.state.vehicle_speed_kmh, 0.1) == 120.0
    assert pytest.approx(gw.state.steering_angle_deg, 0.1) == 0.0


def test_lin_bcm_ve_someip_rpc():
    """LIN BCM kapı kilidi ve SOME/IP Otopilot RPC köprüleri test edilir."""
    gw = TeslaCentralGateway()

    # LIN BCM: Kilitli (0x01)
    gw.decode_lin_bcm(0x24, bytes([0x01, 0x00]))
    assert gw.state.door_lock_status is True

    # SOME/IP: FSD Otopilot Aktifleştir (Payload: 0x01)
    resp = gw.process_someip_rpc(0x1234, 0x0001, bytes([0x01]))
    assert resp[0] == 0x00  # E_OK
    assert gw.state.fsd_engaged is True


def test_uds_tehis_sorgulari():
    """UDS 0x22 üzerinden VIN ve güç verilerinin doğru çekildiği test edilir."""
    gw = TeslaCentralGateway()
    gw.state.pack_voltage_v = 400.0
    gw.state.pack_current_a = 200.0
    gw.state.power_kw = 80.0  # 80 kW

    # DID 0xF190 (VIN)
    resp_vin = gw.handle_uds_request(bytes([0x22, 0xF1, 0x90]))
    assert resp_vin[:3] == bytes([0x62, 0xF1, 0x90])
    assert resp_vin[3:] == b"5YJ3E1EB8NF123456"

    # DID 0x0104 (Power kW)
    resp_pwr = gw.handle_uds_request(bytes([0x22, 0x01, 0x04]))
    assert resp_pwr[:3] == bytes([0x62, 0x01, 0x04])
    pwr_val = struct.unpack(">f", resp_pwr[3:])[0]
    assert pytest.approx(pwr_val, 0.1) == 80.0
