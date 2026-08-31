"""
Tesla C++20 Kavramlar ve Meta-Programlama Birim Testleri (PyTest)
================================================================
Bu test paketi; C++20 Concepts kurallarini, requires kisitlamalarini
ve constexpr CRC32 serilestirme guvenligini dogrular.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import pytest
import sys
import os
import time

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_kavramlar_meta import (
    ConstexprCRC32,
    TeslaSensorPaketiKavrami,
    TeslaBataryaTelemetrisi,
    TeslaMotorTelemetrisi,
    GecersizPaketOrnegi,
    TeslaTipGuvenliSerilestirici
)


def test_constexpr_crc32_dogrulugu():
    """CRC32 algoritmasının standart test vektörleriyle doğruluğu test edilir."""
    veri = b"123456789"
    beklenen_crc = 0xCBF43926
    hesaplanan = ConstexprCRC32.hesapla(veri)
    assert hesaplanan == beklenen_crc, f"CRC32 Hatası: Beklenen 0x{beklenen_crc:X}, Çıkan 0x{hesaplanan:X}"


def test_tesla_sensor_paketi_kavrami_gecerli_turler():
    """Geçerli POD telemetri yapıları concept doğrulamasından geçmelidir."""
    batarya = TeslaBataryaTelemetrisi(0x100, time.time_ns(), 400.0, 10.0, 25.0, 90.0)
    motor = TeslaMotorTelemetrisi(0x200, time.time_ns(), 3000.0, 250.0, 248.0, 65.0)

    b_gecerli, b_msg = TeslaSensorPaketiKavrami.dogrula(batarya)
    m_gecerli, m_msg = TeslaSensorPaketiKavrami.dogrula(motor)

    assert b_gecerli is True
    assert m_gecerli is True


def test_tesla_sensor_paketi_kavrami_gecersiz_tur_reddi():
    """Requires kısıtlamalarını sağlamayan türler reddedilmelidir."""
    gecersiz = GecersizPaketOrnegi(veri_metni="HATA", deger=12.5)
    gecerli_mi, hata_mesaji = TeslaSensorPaketiKavrami.dogrula(gecersiz)

    assert gecerli_mi is False
    assert "HATA" in hata_mesaji


def test_tip_guvenli_serilestirici_basarili():
    """Uyumlu paketler serileştirilmeli ve sonuna 4 baytlık CRC32 eklenmelidir."""
    serilestirici = TeslaTipGuvenliSerilestirici()
    batarya = TeslaBataryaTelemetrisi(0x140, 1000, 399.5, -50.0, 28.0, 80.0)
    
    cikti = serilestirici.serilestir_ve_crc_ekle(batarya)
    ham_govde = batarya.baytlara_donustur()
    
    assert len(cikti) == len(ham_govde) + 4  # Gövde + 4 bayt CRC32
    crc_cikti = int.from_bytes(cikti[-4:], byteorder="little")
    assert crc_cikti == ConstexprCRC32.hesapla(ham_govde)


def test_tip_guvenli_serilestirici_tip_hatasi_firlatma():
    """Concept kurallarını ihlal eden nesneler TypeError fırlatmalıdır."""
    serilestirici = TeslaTipGuvenliSerilestirici()
    gecersiz = GecersizPaketOrnegi("HATALI", 0.0)

    with pytest.raises(TypeError) as exc_info:
        serilestirici.serilestir_ve_crc_ekle(gecersiz)
    
    assert "Concept Constraint Hatasi" in str(exc_info.value)
