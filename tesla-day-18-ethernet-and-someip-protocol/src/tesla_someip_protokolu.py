"""
Tesla Ethernet ve SOME/IP Protokol Modulu
==========================================
Bu modul; AUTOSAR / ISO standartlarindaki SOME/IP (Scalable service-Oriented
MiddlewarE over IP) 16-byte baslik yapisini, SOME/IP-SD (Service Discovery)
mekanizmasini, RPC (Request-Response) ve Publish/Subscribe modellerini gercekler.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum
import struct
import time


class SOMEIPMessageType(IntEnum):
    REQUEST = 0x00
    REQUEST_NO_RETURN = 0x01
    NOTIFICATION = 0x02
    RESPONSE = 0x80
    ERROR = 0x81


class SOMEIPReturnCode(IntEnum):
    E_OK = 0x00
    E_NOT_OK = 0x01
    E_UNKNOWN_SERVICE = 0x02
    E_UNKNOWN_METHOD = 0x03
    E_NOT_READY = 0x04


@dataclass
class TeslaSOMEIPHeader:
    service_id: int        # 16-bit
    method_id: int         # 16-bit (Event için 1-bit event bayrağı set edilir)
    uzunluk: int           # 32-bit (Request ID'den payload sonuna kadar olan byte sayısı)
    client_id: int         # 16-bit
    session_id: int        # 16-bit
    protocol_version: int = 0x01   # 8-bit
    interface_version: int = 0x01  # 8-bit
    message_type: SOMEIPMessageType = SOMEIPMessageType.REQUEST
    return_code: SOMEIPReturnCode = SOMEIPReturnCode.E_OK

    def serilestir(self) -> bytes:
        message_id = (self.service_id << 16) | (self.method_id & 0xFFFF)
        request_id = (self.client_id << 16) | (self.session_id & 0xFFFF)
        return struct.pack(
            ">IIIBBBB",
            message_id,
            self.uzunluk,
            request_id,
            self.protocol_version,
            self.interface_version,
            int(self.message_type),
            int(self.return_code)
        )

    @classmethod
    def ayristir(cls, ham_veri: bytes) -> 'TeslaSOMEIPHeader':
        message_id, uzunluk, request_id, proto_ver, iface_ver, msg_type, ret_code = struct.unpack(
            ">IIIBBBB",
            ham_veri[:16]
        )
        return cls(
            service_id=(message_id >> 16) & 0xFFFF,
            method_id=message_id & 0xFFFF,
            uzunluk=uzunluk,
            client_id=(request_id >> 16) & 0xFFFF,
            session_id=request_id & 0xFFFF,
            protocol_version=proto_ver,
            interface_version=iface_ver,
            message_type=SOMEIPMessageType(msg_type),
            return_code=SOMEIPReturnCode(ret_code)
        )


@dataclass
class TeslaSOMEIPPaket:
    baslik: TeslaSOMEIPHeader
    payload: bytes

    def ikiliye_donustur(self) -> bytes:
        self.baslik.uzunluk = 8 + len(self.payload)  # 8 byte (Request ID'den sonrasının uzunluğu)
        return self.baslik.serilestir() + self.payload

    @classmethod
    def ikiliden_coz(cls, ham_veri: bytes) -> 'TeslaSOMEIPPaket':
        baslik = TeslaSOMEIPHeader.ayristir(ham_veri)
        payload = ham_veri[16: 16 + (baslik.uzunluk - 8)]
        return cls(baslik=baslik, payload=payload)


class TeslaSOMEIPServer:
    """
    Tesla Otopilot Hedef Hız Sunucusu (SOME/IP Service ID = 0x1234).
    Method ID = 0x0001: Hedef Hız Ayarla (float km/h)
    """
    def __init__(self, service_id: int = 0x1234):
        self.service_id = service_id
        self.mevcut_hedef_hiz_kmh = 0.0

    def istek_isle(self, gelen_paket: TeslaSOMEIPPaket) -> TeslaSOMEIPPaket:
        # Servis ve Metot Kontrolü
        if gelen_paket.baslik.service_id != self.service_id:
            yanit_baslik = TeslaSOMEIPHeader(
                service_id=gelen_paket.baslik.service_id,
                method_id=gelen_paket.baslik.method_id,
                uzunluk=8,
                client_id=gelen_paket.baslik.client_id,
                session_id=gelen_paket.baslik.session_id,
                message_type=SOMEIPMessageType.ERROR,
                return_code=SOMEIPReturnCode.E_UNKNOWN_SERVICE
            )
            return TeslaSOMEIPPaket(baslik=yanit_baslik, payload=b'')

        if gelen_paket.baslik.method_id == 0x0001:
            # Hedef Hız Güncelleme (4-byte float payload)
            if len(gelen_paket.payload) >= 4:
                yeni_hiz = struct.unpack(">f", gelen_paket.payload[:4])[0]
                self.mevcut_hedef_hiz_kmh = yeni_hiz
                
                yanit_baslik = TeslaSOMEIPHeader(
                    service_id=self.service_id,
                    method_id=0x0001,
                    uzunluk=8 + 4,
                    client_id=gelen_paket.baslik.client_id,
                    session_id=gelen_paket.baslik.session_id,
                    message_type=SOMEIPMessageType.RESPONSE,
                    return_code=SOMEIPReturnCode.E_OK
                )
                yanit_payload = struct.pack(">f", self.mevcut_hedef_hiz_kmh)
                return TeslaSOMEIPPaket(baslik=yanit_baslik, payload=yanit_payload)

        # Bilinmeyen Metot
        yanit_baslik = TeslaSOMEIPHeader(
            service_id=self.service_id,
            method_id=gelen_paket.baslik.method_id,
            uzunluk=8,
            client_id=gelen_paket.baslik.client_id,
            session_id=gelen_paket.baslik.session_id,
            message_type=SOMEIPMessageType.ERROR,
            return_code=SOMEIPReturnCode.E_UNKNOWN_METHOD
        )
        return TeslaSOMEIPPaket(baslik=yanit_baslik, payload=b'')


class TeslaSOMEIPClient:
    """
    Tesla SOME/IP İstemcisi (Otopilot UI veya Planlayıcı).
    """
    def __init__(self, client_id: int = 0x0042):
        self.client_id = client_id
        self.session_sayaci = 1

    def rpc_hedef_hiz_cagir(self, server: TeslaSOMEIPServer, hiz_kmh: float) -> Tuple[bool, float]:
        istek_baslik = TeslaSOMEIPHeader(
            service_id=server.service_id,
            method_id=0x0001,
            uzunluk=8 + 4,
            client_id=self.client_id,
            session_id=self.session_sayaci,
            message_type=SOMEIPMessageType.REQUEST,
            return_code=SOMEIPReturnCode.E_OK
        )
        self.session_sayaci += 1
        payload = struct.pack(">f", hiz_kmh)
        istek_paketi = TeslaSOMEIPPaket(baslik=istek_baslik, payload=payload)

        # Ağ üzerinden iletim simülasyonu
        ham_istek = istek_paketi.ikiliye_donustur()
        gelen_istek = TeslaSOMEIPPaket.ikiliden_coz(ham_istek)
        
        yanit_paketi = server.istek_isle(gelen_istek)
        ham_yanit = yanit_paketi.ikiliye_donustur()
        cozulen_yanit = TeslaSOMEIPPaket.ikiliden_coz(ham_yanit)

        if cozulen_yanit.baslik.return_code == SOMEIPReturnCode.E_OK and len(cozulen_yanit.payload) >= 4:
            onaylanan_hiz = struct.unpack(">f", cozulen_yanit.payload[:4])[0]
            return True, onaylanan_hiz
        return False, 0.0
