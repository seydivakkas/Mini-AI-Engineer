"""
Tesla Linux epoll Birim Testleri (PyTest)
=========================================
Bu test paketi; Linux epoll API semantigini, EPOLLET Edge-Triggered bildirimlerini,
Level-Triggered durumlarini ve eventfd sinyallesmesini dogrular.

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

from src.tesla_epoll_reaktoru import (
    TeslaEpollOlayReaktoru,
    EpollTetiklemeModu,
    EpollOlayTipi,
    TeslaOlayFd
)


def test_epoll_ctl_ekle_ve_sil():
    """Dosya tanımlayıcı ekleme ve silme operasyonları doğrulanır."""
    reaktor = TeslaEpollOlayReaktoru()
    assert reaktor.epoll_ctl_ekle(fd_id=10, olay_maskesi=EpollOlayTipi.EPOLLIN, kullanici_verisi="kamera_0") is True
    assert reaktor.epoll_ctl_ekle(fd_id=10, olay_maskesi=EpollOlayTipi.EPOLLIN, kullanici_verisi="tekrar") is False  # Çift ekleme engellenmeli
    assert reaktor.epoll_ctl_sil(fd_id=10) is True
    assert reaktor.epoll_ctl_sil(fd_id=10) is False


def test_epollet_edge_triggered_tek_bildirim():
    """Edge-Triggered (EPOLLET) modunda yeni veri geldiğinde sadece 1 kez bildirim yapıldığı test edilir."""
    reaktor = TeslaEpollOlayReaktoru()
    reaktor.epoll_ctl_ekle(fd_id=1, olay_maskesi=EpollOlayTipi.EPOLLIN, kullanici_verisi="fsd_kamera",
                           tetikleme=EpollTetiklemeModu.EDGE_TRIGGERED_EPOLLET)

    # 1. Veri geldi -> Bildirim olmalı
    reaktor.veri_geldi_sinyali(fd_id=1, bayt_sayisi=1024)
    olaylar1 = reaktor.epoll_wait(maks_olay=10)
    assert len(olaylar1) == 1
    assert olaylar1[0]["fd_id"] == 1

    # 2. Tampon hala dolu olsa bile yeni veri gelmediği için EPOLLET 2. kez tetiklenmemeli
    olaylar2 = reaktor.epoll_wait(maks_olay=10)
    assert len(olaylar2) == 0


def test_level_triggered_surekli_bildirim():
    """Level-Triggered modunda tampon boşalana kadar sürekli bildirim yapıldığı test edilir."""
    reaktor = TeslaEpollOlayReaktoru()
    reaktor.epoll_ctl_ekle(fd_id=2, olay_maskesi=EpollOlayTipi.EPOLLIN, kullanici_verisi="can_bus",
                           tetikleme=EpollTetiklemeModu.LEVEL_TRIGGERED)

    reaktor.veri_geldi_sinyali(fd_id=2, bayt_sayisi=128)
    olaylar1 = reaktor.epoll_wait(maks_olay=10)
    assert len(olaylar1) == 1

    # Tampon boşaltılmadı -> Tekrar tetiklenmeli
    olaylar2 = reaktor.epoll_wait(maks_olay=10)
    assert len(olaylar2) == 1

    # Tamponu boşaltınca bildirim kesilmeli
    reaktor.tamponu_bosalt_tuket(fd_id=2)
    olaylar3 = reaktor.epoll_wait(maks_olay=10)
    assert len(olaylar3) == 0


def test_eventfd_sinyallesme():
    """eventfd hafif sinyalleşme sayacının doğru çalıştığı test edilir."""
    efd = TeslaOlayFd()
    assert efd.sinyal_oku() == 0
    efd.sinyal_yaz(5)
    efd.sinyal_yaz(3)
    assert efd.sinyal_oku() == 8
    assert efd.sinyal_oku() == 0  # Okunduktan sonra sıfırlanmalı
