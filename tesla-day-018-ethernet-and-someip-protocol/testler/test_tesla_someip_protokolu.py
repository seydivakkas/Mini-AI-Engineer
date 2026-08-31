"""
Tesla Ethernet ve SOME/IP Birim Testleri (PyTest)
=================================================
Bu test paketi; SOME/IP 16-byte baslik paketlemesini, RPC Request-Response
akisini, hedef hiz guncellemesini ve hata yonetimini dogrular.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import pytest
import struct
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_someip_protokolu import (
    TeslaSOMEIPHeader,
    TeslaSOMEIPPaket,
    TeslaSOMEIPServer,
    TeslaSOMEIPClient,
    SOMEIPMessageType,
    SOMEIPReturnCode
)


def test_someip_header_serilestirme_ve_cozme():
    """16-byte SOME/IP başlığının ikiliye dönüştürülüp kayıpsız geri çözüldüğü test edilir."""
    h = TeslaSOMEIPHeader(
        service_id=0x1234,
        method_id=0x0001,
        uzunluk=12,
        client_id=0x0042,
        session_id=0x0005,
        protocol_version=0x01,
        interface_version=0x01,
        message_type=SOMEIPMessageType.REQUEST,
        return_code=SOMEIPReturnCode.E_OK
    )
    raw = h.serilestir()
    assert len(raw) == 16

    cozulmus = TeslaSOMEIPHeader.ayristir(raw)
    assert cozulmus.service_id == 0x1234
    assert cozulmus.method_id == 0x0001
    assert cozulmus.client_id == 0x0042
    assert cozulmus.session_id == 0x0005
    assert cozulmus.message_type == SOMEIPMessageType.REQUEST
    assert cozulmus.return_code == SOMEIPReturnCode.E_OK


def test_someip_rpc_basarili_cagri():
    """İstemcinin Sunucuya RPC çağrısı yaparak hedef hızı 130 km/h olarak güncellediği test edilir."""
    server = TeslaSOMEIPServer(service_id=0x1234)
    client = TeslaSOMEIPClient(client_id=0x0042)

    basarili, onaylanan_hiz = client.rpc_hedef_hiz_cagir(server, 130.0)

    assert basarili is True
    assert pytest.approx(onaylanan_hiz, 0.01) == 130.0
    assert pytest.approx(server.mevcut_hedef_hiz_kmh, 0.01) == 130.0


def test_someip_bilinmeyen_servis_hatasi():
    """Var olmayan bir servis ID'si çağrıldığında E_UNKNOWN_SERVICE hatası alındığı test edilir."""
    server = TeslaSOMEIPServer(service_id=0x1234)
    
    gecersiz_baslik = TeslaSOMEIPHeader(
        service_id=0x9999,  # Yanlış servis
        method_id=0x0001,
        uzunluk=8,
        client_id=0x0042,
        session_id=1,
        message_type=SOMEIPMessageType.REQUEST,
        return_code=SOMEIPReturnCode.E_OK
    )
    paket = TeslaSOMEIPPaket(baslik=gecersiz_baslik, payload=b'')
    yanit = server.istek_isle(paket)

    assert yanit.baslik.message_type == SOMEIPMessageType.ERROR
    assert yanit.baslik.return_code == SOMEIPReturnCode.E_UNKNOWN_SERVICE


def test_someip_bilinmeyen_metot_hatasi():
    """Var olmayan bir metot ID'si çağrıldığında E_UNKNOWN_METHOD hatası alındığı test edilir."""
    server = TeslaSOMEIPServer(service_id=0x1234)
    
    gecersiz_baslik = TeslaSOMEIPHeader(
        service_id=0x1234,
        method_id=0xFFFF,  # Yanlış metot
        uzunluk=8,
        client_id=0x0042,
        session_id=1,
        message_type=SOMEIPMessageType.REQUEST,
        return_code=SOMEIPReturnCode.E_OK
    )
    paket = TeslaSOMEIPPaket(baslik=gecersiz_baslik, payload=b'')
    yanit = server.istek_isle(paket)

    assert yanit.baslik.message_type == SOMEIPMessageType.ERROR
    assert yanit.baslik.return_code == SOMEIPReturnCode.E_UNKNOWN_METHOD
