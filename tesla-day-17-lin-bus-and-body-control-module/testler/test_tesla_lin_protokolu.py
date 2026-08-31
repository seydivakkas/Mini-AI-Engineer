"""
Tesla LIN ve BCM Birim Testleri (PyTest)
========================================
Bu test paketi; PID parite bitlerini (P0, P1), LIN 2.x Gelismis Checksum'i
ve BCM govde kontrolorunun (Pencere, Koltuk) durum guncellemelerini dogrular.

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

from src.tesla_lin_protokolu import (
    TeslaLINSlaveBCM,
    TeslaLINMasterCizelgeleyici,
    TeslaLINMesaj,
    pid_hesapla,
    pid_dogrula,
    gelismis_checksum_hesapla
)


def test_pid_parite_hesaplama_ve_dogrulama():
    """6-bit Frame ID'den türetilen 8-bit PID'nin parite doğrulamadan başarıyla geçtiği test edilir."""
    frame_id = 0x32  # 50
    pid = pid_hesapla(frame_id)

    assert (pid & 0x3F) == 0x32
    assert pid_dogrula(pid) is True


def test_gelismis_checksum_hesaplama():
    """LIN 2.x Enhanced Checksum hesaplamasının terslenmiş elde toplamını doğru ürettiği test edilir."""
    pid = pid_hesapla(0x32)
    veri = b'\x50'
    csum = gelismis_checksum_hesapla(pid, veri)

    assert 0 <= csum <= 255
    # Çözücüde toplam + csum == 0xFF olmalıdır
    toplam = pid + veri[0]
    if toplam > 0xFF:
        toplam = (toplam & 0xFF) + 1
    assert ((toplam + csum) & 0xFF) == 0xFF


def test_bcm_pencere_ve_koltuk_kontrolu():
    """BCM modülünün geçerli LIN paketleriyle pencere ve koltuk konumlarını güncellediği test edilir."""
    bcm = TeslaLINSlaveBCM()
    master = TeslaLINMasterCizelgeleyici(bcm)

    # 1. Pencere %75 Açma
    msg_cam = master.cerceve_gonder(0x32, b'\x4B')  # 75
    yanit_cam = bcm.lin_mesaj_isle(msg_cam)
    assert bcm.pencere_seviyesi_yuzde == 75.0
    assert yanit_cam["aygit"] == "Pencere"

    # 2. Koltuk 180 mm Ayarlama
    msg_koltuk = master.cerceve_gonder(0x14, b'\x5A')  # 90 * 2 = 180 mm
    yanit_koltuk = bcm.lin_mesaj_isle(msg_koltuk)
    assert bcm.koltuk_pozisyonu_mm == 180.0
    assert yanit_koltuk["aygit"] == "Koltuk"


def test_gecersiz_pid_reddi():
    """Bozuk pariteye sahip sahte PID'lerin reddedildiği test edilir."""
    sahte_pid = 0x32 ^ 0x40  # P0 biti ters çevrilmiş
    assert pid_dogrula(sahte_pid) is False
