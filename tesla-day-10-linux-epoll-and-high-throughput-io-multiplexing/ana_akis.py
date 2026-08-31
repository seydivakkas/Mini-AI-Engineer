"""
Tesla Gun 10 Ana Akis (Tesla Day 10 Main Pipeline)
===================================================
Linux epoll O(1) Reaktor Dongusu, EPOLLET ve Yüksek Hizli Coklama
Uctan Uca Calistirma ve Teshis Paneli Uretim Scripti.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
if su_an_dizin not in sys.path:
    sys.path.insert(0, su_an_dizin)

from src.tesla_epoll_reaktoru import (
    TeslaEpollOlayReaktoru,
    EpollTetiklemeModu,
    EpollOlayTipi,
    TeslaOlayFd
)
from src.tesla_epoll_profilleyici import TeslaEpollProfilleyici
from src.tesla_epoll_gorsellestirici import TeslaEpollGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GOMULU YAZILIM MASTERI | GUN 10: LINUX EPOLL & I/O MULTIPLEXING 🚗")
    print("================================================================================")
    print("Stajyer Gorevi: 8 Kamera + 4 CAN Soketini EPOLLET ile Tek Reaktor Dongusunde Yonet")
    print("--------------------------------------------------------------------------------\n")

    # 1. epoll Reaktörü Kurulumu
    print(" [1] Tesla FSD I/O Reaktörü Başlatılıyor...")
    reaktor = TeslaEpollOlayReaktoru()

    # 8 Kamera Akışı Kaydı
    for cam_id in range(8):
        reaktor.epoll_ctl_ekle(
            fd_id=100 + cam_id,
            olay_maskesi=EpollOlayTipi.EPOLLIN,
            kullanici_verisi=f"Kamera_{cam_id}_Surround",
            tetikleme=EpollTetiklemeModu.EDGE_TRIGGERED_EPOLLET
        )

    # 4 CAN Veri Yolu Kaydı
    for can_id in range(4):
        reaktor.epoll_ctl_ekle(
            fd_id=200 + can_id,
            olay_maskesi=EpollOlayTipi.EPOLLIN,
            kullanici_verisi=f"CAN_Bus_{can_id}",
            tetikleme=EpollTetiklemeModu.EDGE_TRIGGERED_EPOLLET
        )

    print(f"     -> 12 Donanımsal Soket (8 Kamera + 4 CAN) epoll Ağacına Eklendi (EPOLLET).")

    # 2. Eşzamanlı Veri Girişi Simülasyonu
    print("\n [2] Kamera 0 (Ön Ana), Kamera 3 (Geri) ve CAN 0 (BMS) Veri Girişi...")
    reaktor.veri_geldi_sinyali(fd_id=100, bayt_sayisi=3110400) # 1080p frame
    reaktor.veri_geldi_sinyali(fd_id=103, bayt_sayisi=3110400)
    reaktor.veri_geldi_sinyali(fd_id=200, bayt_sayisi=16)      # CAN frame

    tetiklenenler = reaktor.epoll_wait(maks_olay=16)
    print(f"     -> [epoll_wait] Tetiklenen Olay Sayısı: {len(tetiklenenler)}")
    for olay in tetiklenenler:
        print(f"        * Soket FD: {olay['fd_id']} -> {olay['veri']} ({olay['tampon_boyutu']:,} Bayt)")

    # 3. Profilleme ve Ölçeklenme Analizi
    print("\n [3] epoll O(1) vs select/poll O(N) Benchmark'ı (2000 Sokete Kadar)...")
    profilleyici = TeslaEpollProfilleyici()
    metrikler = profilleyici.benchmark_olceklenme_analizi()

    print(f"     -> epoll Ortalama Gecikme         : {metrikler['epoll_ortalama_us']:.3f} µs (Sabit O(1))")
    print(f"     -> select/poll Ortalama Gecikme   : {metrikler['select_ortalama_us']:.3f} µs (Doğrusal O(N))")
    print(f"     -> 2000 Sokette Hızlanma Çarpanı  : {metrikler['maksimum_hizlanma']:.1f}x Daha Hızlı")

    # 4. Tanı Paneli Görselleştirme
    print("\n [4] 6 Panelli Tesla epoll Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaEpollGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_epoll_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 10 BAŞARIYLA TAMAMLANDI! LINUX EPOLL REAKTÖRÜ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
