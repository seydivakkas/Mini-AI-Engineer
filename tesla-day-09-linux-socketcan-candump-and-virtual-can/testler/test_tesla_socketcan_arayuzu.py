"""
Tesla Linux SocketCAN Birim Testleri (PyTest)
=============================================
Bu test paketi; Linux struct can_frame serilestirmesini, CAN_RAW_FILTER
donanim maskelemesini ve vcan0 sanal ag simulasyonunu dogrular.

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

from src.tesla_socketcan_arayuzu import (
    TeslaCANFrame,
    TeslaCANFiltresi,
    TeslaSocketCANArayuzu,
    TeslaVCanAgSimulatoru
)


def test_can_frame_ikili_serilestirme():
    """Linux 16-baytlık struct can_frame yapısına tam uyumlu paketleme doğrulanır."""
    orijinal = TeslaCANFrame(can_id=0x123, can_dlc=4, data=b'\xDE\xAD\xBE\xEF')
    baytlar = orijinal.to_bytes()

    assert len(baytlar) == 16  # Standart Linux can_frame boyutu
    cozulmus = TeslaCANFrame.from_bytes(baytlar)

    assert cozulmus.can_id == 0x123
    assert cozulmus.can_dlc == 4
    assert cozulmus.data == b'\xDE\xAD\xBE\xEF'


def test_can_filtresi_eslesme():
    """Maske eşleşme kuralları doğrulanır."""
    filtre = TeslaCANFiltresi(can_id=0x100, can_mask=0x700)
    
    assert filtre.eslesiyor_mu(0x100) is True
    assert filtre.eslesiyor_mu(0x150) is True  # 0x150 & 0x700 == 0x100
    assert filtre.eslesiyor_mu(0x200) is False # 0x200 & 0x700 != 0x100


def test_socketcan_kernel_filtreleme():
    """Filtreye uyan frame'lerin kuyruğa girdiği, uymayanların düşürüldüğü test edilir."""
    soket = TeslaSocketCANArayuzu("vcan0")
    soket.baglan()
    soket.filtre_ekle(can_id=0x100, can_mask=0x7FF)

    frame_gecerli = TeslaCANFrame(can_id=0x100, can_dlc=2, data=b'\x01\x02')
    frame_gecersiz = TeslaCANFrame(can_id=0x200, can_dlc=2, data=b'\x03\x04')

    assert soket.kernel_filtresinden_gecir_ve_kabul_et(frame_gecerli) is True
    assert soket.kernel_filtresinden_gecir_ve_kabul_et(frame_gecersiz) is False

    alinan = soket.frame_al()
    assert alinan is not None
    assert alinan.can_id == 0x100
    assert soket.frame_al() is None


def test_vcan_ag_simulatoru_broadcast():
    """vcan0 ağında yayınlanan paketin diğer düğümlere filtre kurallarıyla ulaştığı test edilir."""
    ag = TeslaVCanAgSimulatoru()
    
    bms = TeslaSocketCANArayuzu("vcan0")
    motor = TeslaSocketCANArayuzu("vcan0")
    fsd = TeslaSocketCANArayuzu("vcan0")

    # FSD sadece 0x100 (BMS) ve 0x200 (Motor) dinliyor
    fsd.filtre_ekle(0x100, 0x7FF)
    fsd.filtre_ekle(0x200, 0x7FF)

    ag.dugum_ekle(bms)
    ag.dugum_ekle(motor)
    ag.dugum_ekle(fsd)

    # BMS paket gönderiyor
    bms_frame = TeslaCANFrame(can_id=0x100, can_dlc=4, data=b'\x00\x00\x01\x90')
    ag.yayinla(bms, bms_frame)

    alinan_fsd = fsd.frame_al()
    assert alinan_fsd is not None
    assert alinan_fsd.can_id == 0x100
