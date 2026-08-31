"""
Tesla CAN-FD Parser ve CRC Birim Testleri (PyTest)
==================================================
Bu test paketi; CRC-17 (<=16B), CRC-21 (>16B) hesaplamalarini,
ikili akis serilestirme/ayristirma ve bit-flip hata tespitini dogrular.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import pytest
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_can_fd_parser import (
    TeslaCANFDFrameParser,
    hesapla_crc17,
    hesapla_crc21
)


def test_crc17_dogrulama_kisa_payload():
    """16 byte ve altı payload'lar için CRC-17'nin doğru hesaplandığı ve doğrulandığı test edilir."""
    parser = TeslaCANFDFrameParser()
    veri_16b = b'BATTERY_TEMP_048'
    
    paket = parser.cerceve_serilestir(can_id=0x120, veri=veri_16b)
    ayrismis = parser.cerceve_ayristir(paket)

    assert ayrismis.gecerli_mi is True
    assert ayrismis.crc_turu == "CRC-17"
    assert ayrismis.dlc == 16
    assert ayrismis.can_id == 0x120
    assert ayrismis.hata_kodu == "TAMAM"


def test_crc21_dogrulama_uzun_payload():
    """16 byte üzeri (64 byte) payload'lar için CRC-21'in doğru çalıştığı test edilir."""
    parser = TeslaCANFDFrameParser()
    veri_64b = b'T' * 64
    
    paket = parser.cerceve_serilestir(can_id=0x380, veri=veri_64b)
    ayrismis = parser.cerceve_ayristir(paket)

    assert ayrismis.gecerli_mi is True
    assert ayrismis.crc_turu == "CRC-21"
    assert ayrismis.dlc == 64
    assert ayrismis.can_id == 0x380


def test_bozuk_bit_crc_hatasi_yakalama():
    """Veri akışında 1-bit bile bozulsa CRC_ERROR_BIT_FLIP hatası üretildiği test edilir."""
    parser = TeslaCANFDFrameParser()
    veri = b'CRITICAL_STEERING_TORQUE_COMMAND'
    
    paket_temiz = parser.cerceve_serilestir(can_id=0x010, veri=veri)
    
    # 1-bit ters çevirme (Bit-flip)
    paket_bozuk = bytearray(paket_temiz)
    paket_bozuk[8] ^= 0x01
    
    ayrismis = parser.cerceve_ayristir(bytes(paket_bozuk))
    assert ayrismis.gecerli_mi is False
    assert ayrismis.hata_kodu == "CRC_ERROR_BIT_FLIP"
    assert ayrismis.alinan_crc != ayrismis.hesaplanan_crc


def test_gecersiz_kisa_boyut_hatasi():
    """8 byte'tan kısa geçersiz veri akışında boyut hatası üretildiği test edilir."""
    parser = TeslaCANFDFrameParser()
    ayrismis = parser.cerceve_ayristir(b'123')
    assert ayrismis.gecerli_mi is False
    assert ayrismis.hata_kodu == "GECERSIZ_BOYUT_COK_KISA"
