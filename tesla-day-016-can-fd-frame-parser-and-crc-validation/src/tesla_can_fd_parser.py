"""
Tesla CAN-FD Frame Parser ve CRC-17 / CRC-21 Dogrulama Modulu
=============================================================
Bu modul; CAN-FD cerceve anatomisini ayristirir, payload boyutuna gore
CRC-17 (<=16 byte) veya CRC-21 (>16 byte) polinomlarini hesaplar ve
hatali cerceveleri ISO 26262 ASIL-D kurallariyla ayiklar.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import struct


POLINOM_CRC17 = 0x1685B  # x^17 + x^16 + x^14 + x^13 + x^11 + x^6 + x^4 + x^3 + x^1 + 1
POLINOM_CRC21 = 0x302899 # x^21 + x^20 + x^13 + x^11 + x^7 + x^4 + x^3 + 1


def hesapla_crc17(veri: bytes) -> int:
    """CAN-FD 17-bit CRC (Payload <= 16 byte)."""
    crc = 0x0
    for b in veri:
        crc ^= (b << 9)
        for _ in range(8):
            if crc & 0x10000:
                crc = ((crc << 1) ^ POLINOM_CRC17) & 0x1FFFF
            else:
                crc = (crc << 1) & 0x1FFFF
    return crc


def hesapla_crc21(veri: bytes) -> int:
    """CAN-FD 21-bit CRC (Payload > 16 byte)."""
    crc = 0x0
    for b in veri:
        crc ^= (b << 13)
        for _ in range(8):
            if crc & 0x100000:
                crc = ((crc << 1) ^ POLINOM_CRC21) & 0x1FFFFF
            else:
                crc = (crc << 1) & 0x1FFFFF
    return crc


@dataclass
class TeslaCANFDAyrismisFrame:
    can_id: int
    dlc: int
    veri: bytes
    crc_turu: str
    alinan_crc: int
    hesaplanan_crc: int
    gecerli_mi: bool
    hata_kodu: str = "TAMAM"


class TeslaCANFDFrameParser:
    """
    CAN-FD Çerçeve Ayrıştırıcısı ve Doğrulama Motoru.
    """
    def __init__(self):
        self.basarili_sayisi = 0
        self.crc_hatali_sayisi = 0

    def cerceve_serilestir(self, can_id: int, veri: bytes) -> bytes:
        """
        CAN-FD çerçevesini ikili veri akışına paketler:
        [2 Byte CAN ID][1 Byte DLC][N Byte Veri][4 Byte CRC]
        """
        dlc = len(veri)
        if dlc <= 16:
            crc = hesapla_crc17(veri)
            crc_tip_bayt = 17
        else:
            crc = hesapla_crc21(veri)
            crc_tip_bayt = 21

        baslik = struct.pack(">HBB", can_id, dlc, crc_tip_bayt)
        kuyruk = struct.pack(">I", crc)
        return baslik + veri + kuyruk

    def cerceve_ayristir(self, ham_veri: bytes) -> TeslaCANFDAyrismisFrame:
        """
        İkili akıştan CAN-FD çerçevesini çözer ve CRC bütünlüğünü doğrular.
        """
        if len(ham_veri) < 8:
            return TeslaCANFDAyrismisFrame(
                can_id=0, dlc=0, veri=b'', crc_turu="BILINMIYOR",
                alinan_crc=0, hesaplanan_crc=0, gecerli_mi=False,
                hata_kodu="GECERSIZ_BOYUT_COK_KISA"
            )

        can_id, dlc, crc_tip_bayt = struct.unpack(">HBB", ham_veri[:4])
        veri = ham_veri[4: 4 + dlc]
        alinan_crc = struct.unpack(">I", ham_veri[4 + dlc: 4 + dlc + 4])[0]

        if dlc <= 16:
            crc_turu = "CRC-17"
            hesaplanan_crc = hesapla_crc17(veri)
        else:
            crc_turu = "CRC-21"
            hesaplanan_crc = hesapla_crc21(veri)

        gecerli_mi = (alinan_crc == hesaplanan_crc)

        if gecerli_mi:
            self.basarili_sayisi += 1
            hata_kodu = "TAMAM"
        else:
            self.crc_hatali_sayisi += 1
            hata_kodu = "CRC_ERROR_BIT_FLIP"

        return TeslaCANFDAyrismisFrame(
            can_id=can_id,
            dlc=dlc,
            veri=veri,
            crc_turu=crc_turu,
            alinan_crc=alinan_crc,
            hesaplanan_crc=hesaplanan_crc,
            gecerli_mi=gecerli_mi,
            hata_kodu=hata_kodu
        )
