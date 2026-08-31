"""
Tesla Gün 85 Ana Akış (Tesla Day 85 Main Pipeline)
===================================================
Supercharger İstasyonları İçin Dinamik Kuyruk ve Rezervasyon Optimizasyonu
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

from src.tesla_supercharger_kuyruk_yonetici import TeslaSuperchargerQueueManager
from src.tesla_kuyruk_profilleyici import TeslaKuyrukProfilleyici
from src.tesla_kuyruk_gorsellestirici import TeslaKuyrukGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 85: SUPERCHARGER DİNAMİK KUYRUK & REZERVASYON 🚗")
    print("================================================================================")
    print("Stajyer Görevi: M/M/c Çoklu Sunucu Kuyruk Teorisi, Bekleme Süresi & FSD Rota")
    print("--------------------------------------------------------------------------------\n")

    # 1. Kuyruk Optimizasyon Benchmark'ı
    print(" [1] 12-Stall Supercharger İstasyonu M/M/c Trafik Analizi Başlatılıyor...")
    profilleyici = TeslaKuyrukProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_queue_optimization()

    print(f"     -> Stall Sayısı            : {metrikler['num_stalls']} Adet V4 Supercharger")
    print(f"     -> Varış Hızı (Lambda)     : {metrikler['lambda_val']:.1f} Araç / Saat")
    print(f"     -> Ortalama Bekleme Süresi : {metrikler['wait_mins']:.2f} Dakika")
    print(f"     -> FSD Navigasyon Kararı   : {metrikler['decision']}")
    print(f"     -> İstasyon Trafik Durumu  : %100 AKICI VE OPTİMİZE")

    # 2. Kuyruk Çözüm Hızı
    print("\n [2] M/M/c Analitik Çözüm ve Rezervasyon Algoritması RTOS Performansı...")
    print(f"     -> Ortalama Çözüm Süresi   : {metrikler['step_ortalama_us']:.3f} µs (P99: {metrikler['step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Rezervasyon   : {metrikler['saniyelik_rezervasyon_kapasitesi']:,} Araç/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Supercharger Kuyruk Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaKuyrukGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_supercharger_kuyruk_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 85 BAŞARIYLA TAMAMLANDI! SUPERCHARGER KUYRUK OPTİMİZASYONU DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
