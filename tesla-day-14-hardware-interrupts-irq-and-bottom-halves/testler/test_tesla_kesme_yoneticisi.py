"""
Tesla Donanim Kesmeleri ve Bottom-Half Birim Testleri (PyTest)
==============================================================
Bu test paketi; HardIRQ Top-Half ACK bayrak temizligini, Threaded
Bottom-Half AEB radar TTC hesaplamasini ve kesme firtinasi korumasini dogrular.

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

from src.tesla_kesme_yoneticisi import (
    TeslaTopHalfHardIRQ,
    TeslaBottomHalfThreadedIRQ,
    TeslaKesmeFirtinasiOnleyici,
    TeslaKesmeYonetimSistemi
)


def test_tophalf_hardirq_ack_ve_nonblocking():
    """Top-Half kesme işleyicisinin anında ACK bayrağını set ettiği ve IRQ_WAKE_THREAD döndürdüğü test edilir."""
    th = TeslaTopHalfHardIRQ(irq_no=42)
    sonuc = th.kesme_isle()

    assert sonuc == "IRQ_WAKE_THREAD"
    assert th.donanim_ack_bayragi is True
    assert th.top_half_sayaci == 1


def test_bottomhalf_radar_ttc_guvenli_mesafe():
    """Uzak mesafede (TTC > 1.2 s) AEB acil freninin tetiklenmediği test edilir."""
    bh = TeslaBottomHalfThreadedIRQ(ttc_esigi_sn=1.2)
    # 60 metre mesafe, 10 m/s yaklaşma hızı -> TTC = 6.0 s
    sonuc = bh.radar_nokta_bulutu_isle(mesafe_m=60.0, bagil_hiz_mps=-10.0)

    assert sonuc["ttc_sn"] == 6.0
    assert sonuc["acil_fren_tetiklendi"] is False
    assert sonuc["durum"] == "GUVENLI_MESAFE"


def test_bottomhalf_radar_ttc_acil_fren_tetikleme():
    """Kritik mesafede (TTC <= 1.2 s) AEB acil freninin derhal tetiklendiği test edilir."""
    bh = TeslaBottomHalfThreadedIRQ(ttc_esigi_sn=1.2)
    # 15 metre mesafe, 20 m/s yaklaşma hızı -> TTC = 0.75 s
    sonuc = bh.radar_nokta_bulutu_isle(mesafe_m=15.0, bagil_hiz_mps=-20.0)

    assert sonuc["ttc_sn"] == 0.75
    assert sonuc["acil_fren_tetiklendi"] is True
    assert sonuc["durum"] == "KRITIK_TEHLIKE"


def test_kesme_firtinasi_token_bucket_engelleme():
    """Aşırı yüksek kesme frekansında (fırtına) Token-Bucket'ın zararlı kesmeleri engellediği test edilir."""
    firtina = TeslaKesmeFirtinasiOnleyici(maks_irq_hizi_sn=100)
    
    kabul = 0
    red = 0
    for _ in range(500):
        if firtina.kesme_kabul_edilebilir_mi():
            kabul += 1
        else:
            red += 1

    assert kabul <= 105
    assert red > 350
    assert firtina.engellenen_kesme_sayisi == red
