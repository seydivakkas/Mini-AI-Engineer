"""
Tesla Linux Karakter Surucusu Birim Testleri (PyTest)
=====================================================
Bu test paketi; cdev acma/kapama islemlerini, ioctl ASIL-D anahtar dogrulamasini,
copy_from_user bellek guvenligini ve tork sinir kontrollerini dogrular.

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

from src.tesla_karakter_surucusu import (
    TeslaTorkKarakterAygiti,
    TeslaTorkPaketi,
    IOCTL_TESLA_TORK_YAZ,
    IOCTL_TESLA_DURUM_OKU,
    ASIL_D_GUVENLIK_ANAHTARI
)


def test_aygit_acma_ve_kapama():
    """Aygıtın open ve release çağrılarının çalıştığı test edilir."""
    aygit = TeslaTorkKarakterAygiti()
    assert aygit.acik_mi is False
    assert aygit.open() == 0
    assert aygit.acik_mi is True
    assert aygit.release() == 0
    assert aygit.acik_mi is False


def test_gecerli_tork_ioctl_asil_d():
    """Geçerli 0xAA55 anahtarı ile torkun başarıyla ayarlandığı doğrulanır."""
    aygit = TeslaTorkKarakterAygiti()
    aygit.open()

    paket = TeslaTorkPaketi(
        guvenlik_anahtari=ASIL_D_GUVENLIK_ANAHTARI,
        hedef_tork_nm=550.0,
        rejenerasyon_etkin_mi=False
    )
    kod, mesaj = aygit.unlocked_ioctl(IOCTL_TESLA_TORK_YAZ, paket.to_bytes())

    assert kod == 0
    assert aygit.guncel_tork_nm == 550.0
    assert "BAŞARILI" in mesaj


def test_gecersiz_anahtar_reddi():
    """Hatalı güvenlik anahtarı gönderildiğinde torkun reddedildiği test edilir."""
    aygit = TeslaTorkKarakterAygiti()
    aygit.open()

    sahte_paket = TeslaTorkPaketi(
        guvenlik_anahtari=0x1234,  # Hatalı anahtar!
        hedef_tork_nm=300.0,
        rejenerasyon_etkin_mi=False
    )
    kod, mesaj = aygit.unlocked_ioctl(IOCTL_TESLA_TORK_YAZ, sahte_paket.to_bytes())

    assert kod == -1
    assert aygit.guncel_tork_nm == 0.0  # Tork değişmemeli!
    assert "EPERM" in mesaj


def test_tork_limit_asimi_ve_copy_from_user_guvenligi():
    """Limit dışı tork ve eksik bayt gönderildiğinde sistemin korunduğu test edilir."""
    aygit = TeslaTorkKarakterAygiti()
    aygit.open()

    # 1. Limit aşımı (1500 Nm > 1000 Nm)
    asiri_paket = TeslaTorkPaketi(
        guvenlik_anahtari=ASIL_D_GUVENLIK_ANAHTARI,
        hedef_tork_nm=1500.0,
        rejenerasyon_etkin_mi=False
    )
    kod1, _ = aygit.unlocked_ioctl(IOCTL_TESLA_TORK_YAZ, asiri_paket.to_bytes())
    assert kod1 == -22  # EINVAL

    # 2. Bozuk/Eksik bayt (copy_from_user EFAULT)
    kod2, _ = aygit.unlocked_ioctl(IOCTL_TESLA_TORK_YAZ, b"\x01\x02")
    assert kod2 == -14  # EFAULT
