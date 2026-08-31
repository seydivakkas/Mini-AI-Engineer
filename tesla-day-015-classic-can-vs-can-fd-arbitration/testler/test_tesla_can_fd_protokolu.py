"""
Tesla Klasik CAN vs CAN-FD Birim Testleri (PyTest)
==================================================
Bu test paketi; Klasik CAN 8-byte ve CAN-FD 64-byte cerceve yapilarini,
BRS hiz kazancini ve Wired-AND arbitrasyon onceliklerini dogrular.

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

from src.tesla_can_fd_protokolu import (
    TeslaKlasikCANFrame,
    TeslaCANFDFrame,
    TeslaCANArbitrasyonSimulasyonu
)


def test_klasik_can_frame_yapisi():
    """Klasik CAN 2.0B çerçevesinin 8-byte payload ve doğru iletim süresi ürettiği test edilir."""
    f = TeslaKlasikCANFrame(can_id=0x123, veri=b'12345678')
    assert f.dlc == 8
    assert f.iletim_suresi_us_hesapla() > 150.0


def test_can_fd_frame_yapisi_64_byte():
    """CAN-FD çerçevesinin 64-byte payload'u DLC 15 olarak eşleştirdiği ve 5 Mbps BRS kullandığı test edilir."""
    f = TeslaCANFDFrame(can_id=0x123, veri=b'\xAA' * 64, brs_aktif_mi=True)
    assert f.dlc == 15
    assert len(f.veri) == 64
    assert f.iletim_suresi_us_hesapla() < 250.0  # 64 byte olmasına rağmen hızlı iletilir


def test_wired_and_arbitrasyon_en_dusuk_id_kazanir():
    """Aynı anda hatta veri basan düğümlerden en küçük CAN ID'ye sahip olanın (0x010) arbitrasyonu kazandığı test edilir."""
    sim = TeslaCANArbitrasyonSimulasyonu()
    sim.mesaj_ekle("Medya", 0x380, b'DATA', "Bilgi-Eglence")
    sim.mesaj_ekle("AcilFren", 0x010, b'BRAKE', "ASIL-D Acil Fren")
    sim.mesaj_ekle("Motor", 0x120, b'TORQUE', "Sürücü Motoru")

    sonuc = sim.arbitrasyon_yaristir()
    assert sonuc["kazanan"] is not None
    assert sonuc["kazanan"]["can_id"] == 0x010
    assert sonuc["kazanan"]["dugum_adi"] == "AcilFren"
    assert len(sonuc["elenenler"]) == 2


def test_can_fd_bant_genisligi_avantaji():
    """CAN-FD'nin Klasik CAN'a kıyasla en az 8 kat daha fazla veri taşıdığı test edilir."""
    f_klasik = TeslaKlasikCANFrame(can_id=0x100, veri=b'\x00' * 8)
    f_fd = TeslaCANFDFrame(can_id=0x100, veri=b'\x00' * 64, brs_aktif_mi=True)

    assert len(f_fd.veri) == 8 * len(f_klasik.veri)
