"""
Tesla C++20 string_view ve span Birim Testleri (PyTest)
======================================================
Bu test paketi; std::string_view dilimleme mekanizmasini, sifir heap tahsisini
ve NMEA GNSS verilerinin dogru ayristirildigini dogrular.

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

from src.tesla_span_ranges_ayristirici import (
    TeslaStringView,
    TeslaNMEAAyristirici
)


def test_string_view_dilimleme_ve_uzunluk():
    """String view alt dilimlerinin doğru uzunluk ve içerik ürettiği test edilir."""
    ham = "TESLA_MODEL_S_PLAID"
    view = TeslaStringView(ham)
    assert len(view) == len(ham)
    assert str(view) == ham

    alt_view = view.subview(6, 7)  # "MODEL_S"
    assert len(alt_view) == 7
    assert str(alt_view) == "MODEL_S"


def test_nmea_gprmc_gecerli_ayristirma():
    """Standart $GPRMC GPS cümlesinin doğru koordinat ve hıza çözümlendiği test edilir."""
    cumle = "$GPRMC,083559.00,A,3723.2475,N,12208.3845,W,55.4,180.0,300826,,,A*72"
    konum = TeslaNMEAAyristirici.gprmc_ayristir(cumle)

    assert konum is not None
    assert konum.gecerli_mi is True
    assert konum.utc_zamani == "083559.00"
    assert pytest.approx(konum.enlem_derece, 0.0001) == 37.387458
    assert pytest.approx(konum.boylam_derece, 0.0001) == -122.139741
    assert pytest.approx(konum.hiz_kmh, 0.1) == 102.6  # 55.4 knot * 1.852 = 102.6 km/h
    assert konum.rota_acisi_derece == 180.0


def test_nmea_gecersiz_cumle_reddi():
    """Bozuk veya eksik alanlı GPS cümleleri None dönmelidir."""
    assert TeslaNMEAAyristirici.gprmc_ayristir("$GPGGA,123456,37.123,N") is None
    assert TeslaNMEAAyristirici.gprmc_ayristir("GECERSIZ_VERI") is None


def test_string_view_sifir_bellek_tahsisi():
    """Metin parçalamada yeni string listesi yerine string_view üretildiği doğrulanır."""
    metin = "CAN_ID_100,400.2,120.5,35.0"
    view = TeslaStringView(metin)
    parcalar = list(view.parcalara_bol(','))

    assert len(parcalar) == 4
    assert str(parcalar[0]) == "CAN_ID_100"
    assert str(parcalar[1]) == "400.2"
    assert str(parcalar[2]) == "120.5"
    assert str(parcalar[3]) == "35.0"
