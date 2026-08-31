"""
Tesla C++20 Esyordamlar Birim Testleri (PyTest)
==============================================
Bu test paketi; C++20 Coroutine ureticilerini, asenkron gorevleri
ve coklu sensor akis zamanlayicisini dogrular.

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

from src.tesla_esyordam_motoru import (
    EsyordamDurumu,
    TeslaTelemetriUreteci,
    TeslaEsyordamGorevi,
    Tesla10GbpsEthernetHatti
)


def test_esyordam_ureteci_adim_adim_calisma():
    """co_yield üretecinin paketleri doğru sırada ve non-blocking ürettiği test edilir."""
    sensor = TeslaTelemetriUreteci("LIDAR_TEST", toplam_paket=5)
    
    paketler = []
    while True:
        pkt = sensor.siradaki_paketi_al()
        if pkt is None:
            break
        paketler.append(pkt)

    assert len(paketler) == 5
    assert sensor.durum == EsyordamDurumu.TAMAMLANDI
    assert paketler[0].sensor_kaynagi == "LIDAR_TEST"
    assert paketler[4].akis_id == 4


def test_esyordam_gorevi_calisma_dongusu():
    """Asenkron görevin durum geçişleri ve bayt sayımı test edilir."""
    sensor = TeslaTelemetriUreteci("BMS_CAN", toplam_paket=3)
    gorev = TeslaEsyordamGorevi("BMS_GOREVI", sensor)

    assert gorev.durum == EsyordamDurumu.BASKI_ALTINDA
    assert gorev.islenen_paket_sayisi == 0

    # Adım 1
    assert gorev.adim_islet() is True
    assert gorev.islenen_paket_sayisi == 1

    # Adım 2 & 3
    assert gorev.adim_islet() is True
    assert gorev.adim_islet() is True
    assert gorev.islenen_paket_sayisi == 3

    # Tamamlanma
    assert gorev.adim_islet() is False
    assert gorev.durum == EsyordamDurumu.TAMAMLANDI
    assert gorev.toplam_bayt == 3 * 1400


def test_10gbps_ethernet_hatti_coklu_akis():
    """8 sensör akışının kooperatif olarak sıfır kayıpla tüketildiği doğrulanır."""
    hat = Tesla10GbpsEthernetHatti()
    for i in range(4):
        u = TeslaTelemetriUreteci(f"SENSOR_{i}", toplam_paket=10)
        g = TeslaEsyordamGorevi(f"GOREV_{i}", u)
        hat.gorev_ekle(g)

    sonuclar = hat.tum_akis_hatlarini_tukelt()
    assert sonuclar["toplam_bayt"] == 4 * 10 * 1400
    assert sonuclar["toplam_adim"] > 0
