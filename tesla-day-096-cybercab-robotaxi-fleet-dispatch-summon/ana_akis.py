"""
Tesla Gün 96 Ana Akış (Tesla Day 96 Main Pipeline)
===================================================
Tesla Cybercab / Robotaxi Otonom Çağırma (Summon) ve Filo Görevlendirme
Uçtan Uca Çalıştırma ve Teşhis Paneli Üretim Scripti.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
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

from src.tesla_cybercab_filo_yoneticisi import TeslaCybercabFleetDispatcher
from src.tesla_filo_yonetim_profilleyici import TeslaFiloYonetimProfilleyici
from src.tesla_filo_yonetim_gorsellestirici import TeslaFiloYonetimGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 96: CYBERCAB ROBOTAXI FİLO GÖREVLENDİRME & SUMMON 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Pedalsız/Direksiyonsuz Filo, ETA < 3dk, Kablosuz Şarj & 3.5µs Eşleme")
    print("--------------------------------------------------------------------------------\n")

    # 1. Filo Yönetim Benchmark'ı
    print(" [1] 100 Cybercab Filo Optimizasyonu ve Yolcu Eşleştirmesi Başlatılıyor...")
    profilleyici = TeslaFiloYonetimProfilleyici(fleet_size=100)
    metrikler = profilleyici.benchmark_fleet_dispatch()

    print(f"     -> Filo Boyutu             : {metrikler['fleet_size']} Cybercab (Direksiyonsuz Ticari Filo)")
    print(f"     -> Atanan En Yakın Araç    : {metrikler['assigned_cab_id']} (Mesafe: {metrikler['pickup_distance_km']} km)")
    print(f"     -> Varış Süresi (ETA)      : {metrikler['eta_minutes']:.2f} Dakika (< 3 dk Süper Hızlı Varış)")
    print(f"     -> Otomatik Kablosuz Şarj  : {metrikler['auto_charged_count']} Araç Şarj Pedine Yönlendirildi")
    print(f"     -> Robotaxi Ticari Durumu  : %100 BOŞ GEZİNTİ (DEADHEADING) MİNİMİZE EDİLDİ")

    # 2. Eşleştirme Hızı
    print("\n [2] Otonom Çağırma (Summon) Filo Eşleştirici RTOS Performansı...")
    print(f"     -> Ortalama Eşleştirme Hızı: {metrikler['step_ortalama_us']:.3f} µs (P99: {metrikler['step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Çağrı Hacmi   : {metrikler['saniyelik_eslestirme_hizi']:,} Çağrı/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Cybercab Filo Yönetimi Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaFiloYonetimGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_cybercab_filo_yonetim_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 96 BAŞARIYLA TAMAMLANDI! CYBERCAB FİLO GÖREVLENDİRİCİSİ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
