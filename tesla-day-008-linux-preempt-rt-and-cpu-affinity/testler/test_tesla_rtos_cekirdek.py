"""
Tesla Linux PREEMPT_RT Birim Testleri (PyTest)
==============================================
Bu test paketi; RTOS yapilandirmasini, CPU Pinning atamasini,
mlockall bellek kilitlemesini ve 1 kHz kontrol dongusu determinizmini dogrular.

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

from src.tesla_rtos_cekirdek import (
    TeslaGercekZamanliYapilandirici,
    Tesla1kHzKontrolDongusu,
    ZamanlamaPolitikasi
)


def test_rtos_yapilandirma_varsayilanlar():
    """Varsayılan yapılandırmanın Core 3 ve SCHED_FIFO 99 olduğu doğrulanır."""
    yapici = TeslaGercekZamanliYapilandirici(cekirdek_id=3, oncelik=99)
    assert yapici.yapilandirma.hedef_cekirdek_id == 3
    assert yapici.yapilandirma.politika == ZamanlamaPolitikasi.SCHED_FIFO
    assert yapici.yapilandirma.oncelik == 99


def test_cpu_sabitleme_ve_oncelik_atama():
    """CPU çekirdek ataması ve öncelik sınırlandırma (1-99) test edilir."""
    yapici = TeslaGercekZamanliYapilandirici()
    yapici.cpu_sabitle(2)
    assert yapici.yapilandirma.hedef_cekirdek_id == 2

    yapici.oncelik_ata(ZamanlamaPolitikasi.SCHED_RR, oncelik=150)  # Üst sınır 99'a kırpılmalı
    assert yapici.yapilandirma.oncelik == 99
    assert yapici.yapilandirma.politika == ZamanlamaPolitikasi.SCHED_RR


def test_mlockall_etkinlestirme():
    """mlockall çağrısının bellek sayfalarını kilitlediği doğrulanır."""
    yapici = TeslaGercekZamanliYapilandirici()
    assert yapici.yapilandirma.mlockall_etkin_mi is False
    assert yapici.bellek_sayfalarini_kilitle_mlockall() is True
    assert yapici.yapilandirma.mlockall_etkin_mi is True


def test_1khz_kontrol_dongusu_rt_performansi():
    """Hard RT modunda 1 kHz kontrol döngüsünün sıfır deadline kaçırması ile çalıştığı test edilir."""
    yapici = TeslaGercekZamanliYapilandirici(cekirdek_id=3, oncelik=99)
    yapici.bellek_sayfalarini_kilitle_mlockall()

    dongu = Tesla1kHzKontrolDongusu(yapici, hedef_periyot_us=1000.0)
    sonuclar = dongu.donguyu_kos(toplam_tik_sayisi=500)

    assert sonuclar["kacan_deadline_sayisi"] == 0
    assert sonuclar["jitter_standart_sapma_us"] < 5.0
    assert pytest.approx(sonuclar["ortalama_periyot_us"], 1.0) == 1000.0
