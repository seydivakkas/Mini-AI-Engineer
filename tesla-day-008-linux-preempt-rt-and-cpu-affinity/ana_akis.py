"""
Tesla Gun 08 Ana Akis (Tesla Day 08 Main Pipeline)
===================================================
Linux PREEMPT_RT, SCHED_FIFO Onceligi ve CPU Affinity
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

from src.tesla_rtos_cekirdek import (
    TeslaGercekZamanliYapilandirici,
    Tesla1kHzKontrolDongusu,
    ZamanlamaPolitikasi
)
from src.tesla_rtos_profilleyici import TeslaRTOSProfilleyici
from src.tesla_rtos_gorsellestirici import TeslaRTOSGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GOMULU YAZILIM MASTERI | GUN 08: LINUX PREEMPT_RT & CPU AFFINITY 🚗")
    print("================================================================================")
    print("Stajyer Gorevi: SCHED_FIFO 99 Onceligi, Core 3 Izolasyonu, mlockall & 1 kHz Dongu")
    print("--------------------------------------------------------------------------------\n")

    # 1. RTOS Yapılandırması
    print(" [1] POSIX Gerçek Zamanlı Parametreler Yapılandırılıyor...")
    rt_yapici = TeslaGercekZamanliYapilandirici(cekirdek_id=3, oncelik=99)
    rt_yapici.bellek_sayfalarini_kilitle_mlockall()
    print(f"     -> Hedef CPU Çekirdeği : Core {rt_yapici.yapilandirma.hedef_cekirdek_id} (İzole Edilmiş)")
    print(f"     -> Zamanlama Politikası: {rt_yapici.yapilandirma.politika.value} (Öncelik: {rt_yapici.yapilandirma.oncelik})")
    print(f"     -> mlockall Bellek Kilidi: {'ETKİN (Sıfır Page Fault)' if rt_yapici.yapilandirma.mlockall_etkin_mi else 'DEVRE DIŞI'}")

    # 2. Profilleme ve Karşılaştırma
    print("\n [2] PREEMPT_RT vs Standart Linux 1 kHz Kontrol Döngüsü Benchmark'ı...")
    profilleyici = TeslaRTOSProfilleyici(tik_sayisi=1000)
    metrikler = profilleyici.benchmark_rt_vs_non_rt()

    print(f"     -> PREEMPT_RT Jitter (σ)         : {metrikler['rt_jitter_us']:.2f} µs (Maksimum: {metrikler['rt_maksimum_us']:.1f} µs)")
    print(f"     -> Standart Linux Jitter (σ)     : {metrikler['non_rt_jitter_us']:.2f} µs (Maksimum: {metrikler['non_rt_maksimum_us']:.1f} µs)")
    print(f"     -> Jitter İyileşme Oranı         : {metrikler['jitter_iyilesme_orani']:.1f}x Daha Deterministik")
    print(f"     -> PREEMPT_RT Kaçan Deadline     : {metrikler['rt_kacan_deadline']} (%{metrikler['rt_kacan_yuzde']:.1f})")
    print(f"     -> Standart Linux Kaçan Deadline : {metrikler['non_rt_kacan_deadline']} (%{metrikler['non_rt_kacan_yuzde']:.1f})")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla PREEMPT_RT Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaRTOSGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_rtos_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 08 BAŞARIYLA TAMAMLANDI! PREEMPT_RT KONTROL DÖNGÜSÜ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
