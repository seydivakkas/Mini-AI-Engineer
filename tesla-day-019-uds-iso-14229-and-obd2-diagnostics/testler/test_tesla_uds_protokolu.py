"""
Tesla UDS Teşhis ve OBD-II Birim Testleri (PyTest)
==================================================
Bu test paketi; ISO 14229 UDS servislerini, DTC kod çözme algoritmalarını,
Seed-Key güvenlik kilit mekanizmasını ve hata durumlarını doğrular.

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

from src.tesla_uds_protokolu import (
    TeslaUDSServer,
    TeslaUDSClient,
    UDSServiceID,
    UDSNRC,
    DiagnosticSessionType,
    decode_dtc,
    encode_dtc,
    DTCRecord
)


def test_dtc_kod_cozme_ve_paketleme():
    """3 baytlık DTC kodunun doğru çözüldüğü test edilir."""
    raw = bytes([0x0A, 0x1F, 0x00])
    dtc_str = decode_dtc(raw)
    assert dtc_str == "P0A1F-00"

    raw_u = bytes([0xC1, 0x00, 0x87])  # 11 000001 -> U0100
    dtc_u = decode_dtc(raw_u)
    assert dtc_u == "U0100-87"

    encoded = encode_dtc("P", 0x0A1F, 0x16)
    assert len(encoded) == 3
    assert decode_dtc(encoded) == "P0A1F-16"


def test_uds_session_control():
    """Teşhis oturumu geçişleri ve kilit sıfırlama davranışı test edilir."""
    server = TeslaUDSServer()
    client = TeslaUDSClient(server)

    assert server.current_session == DiagnosticSessionType.DEFAULT_SESSION

    ok, session = client.set_session(DiagnosticSessionType.EXTENDED_DIAGNOSTIC_SESSION)
    assert ok is True
    assert session == DiagnosticSessionType.EXTENDED_DIAGNOSTIC_SESSION
    assert server.current_session == DiagnosticSessionType.EXTENDED_DIAGNOSTIC_SESSION


def test_uds_read_did_basarili_ve_hatali():
    """DID okuma ve var olmayan DID sorgulama hatası test edilir."""
    server = TeslaUDSServer()
    client = TeslaUDSClient(server)

    # VIN Okuma (0xF190)
    ok, vin_data = client.read_did(0xF190)
    assert ok is True
    assert vin_data == b"5YJ3E1EB8NF123456"

    # Batarya Voltajı (0x0100)
    ok_v, volt_data = client.read_did(0x0100)
    assert ok_v is True
    voltage = struct.unpack(">f", volt_data)[0]
    assert pytest.approx(voltage, 0.1) == 398.6

    # Geçersiz DID (0xEEEE) -> NRC 0x31 RequestOutOfRange
    ok_inv, inv_data = client.read_did(0xEEEE)
    assert ok_inv is False
    assert inv_data is None


def test_uds_security_access_ve_yazma():
    """Seed-Key güvenlik açma ve güvenli DID yazma akışı test edilir."""
    server = TeslaUDSServer()
    client = TeslaUDSClient(server)

    # Extended Session'a geç
    client.set_session(DiagnosticSessionType.EXTENDED_DIAGNOSTIC_SESSION)

    # Kilit açmadan yazmayı dene -> Başarısız olmalı
    ok_write_before = client.write_did(0x0103, bytes([0x00]))
    assert ok_write_before is False

    # Güvenlik kilidini aç
    sec_unlocked = client.unlock_security()
    assert sec_unlocked is True
    assert server.security_unlocked is True

    # Kilit açıldıktan sonra yaz -> Başarılı olmalı
    ok_write_after = client.write_did(0x0103, bytes([0x00]))
    assert ok_write_after is True
    assert server.did_database[0x0103] == bytes([0x00])


def test_uds_read_ve_clear_dtc():
    """DTC listeleme ve tüm DTC'leri silme (0x14) servisi test edilir."""
    server = TeslaUDSServer()
    client = TeslaUDSClient(server)

    dtcs = client.read_dtcs(0xFF)
    assert len(dtcs) >= 3
    dtc_codes = [d[0] for d in dtcs]
    assert "P0A1F-00" in dtc_codes

    # 0x14 ile temizle
    cleared = client.clear_dtcs()
    assert cleared is True

    # Temizlendikten sonra liste boş olmalı
    dtcs_after = client.read_dtcs(0xFF)
    assert len(dtcs_after) == 0
