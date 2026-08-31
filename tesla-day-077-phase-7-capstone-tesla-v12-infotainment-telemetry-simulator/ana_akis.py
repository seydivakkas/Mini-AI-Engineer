"""
Tesla Gün 77 Ana Akış (Tesla Day 77 Master Capstone Pipeline)
=============================================================
FAZ 7 BÜYÜK CAPSTONE: Tesla V12 Konsol ve Telemetri Simülatörü
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

from src.tesla_v12_full_stack_infotainment_simulator import TeslaV12FullStackInfotainmentSimulator
from src.tesla_capstone_profilleyici import TeslaCapstoneProfilleyici
from src.tesla_capstone_gorsellestirici import TeslaCapstoneGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 77: FAZ 7 BÜYÜK CAPSTONE - V12 İNFOTAINMENT 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Qt6/QML, 3D GPU Render, D-Bus IPC, ARNC, Secure Boot, OTA & HVAC")
    print("--------------------------------------------------------------------------------\n")

    # 1. Capstone Benchmark'ı
    print(" [1] 9 Alt Modüllü Tesla V12 Konsol ve Telemetri Simülasyonu Başlatılıyor...")
    profilleyici = TeslaCapstoneProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_full_stack_infotainment()

    print(f"     -> Araç Telemetrisi        : {metrikler['speed_kmh']:.1f} km/s | Batarya: %{metrikler['battery_pct']:.1f}")
    print(f"     -> 3D GPU Ekran İzdüşümü   : ({metrikler['screen_u']:.1f} px, {metrikler['screen_v']:.1f} px)")
    print(f"     -> PipeWire ARNC Sönümleme : {metrikler['arnc_db']:.1f} dB (Ters Faz Aktif)")
    print(f"     -> UWB Phone Key Mesafesi  : {metrikler['uwb_dist_m']:.2f} m (Işık Hızı Doğrulandı)")
    print(f"     -> Capstone Sistem Sağlığı : {'%100 ALL SYSTEMS GO (MÜKEMMEL ENTEGRASYON)' if metrikler['capstone_ok'] else 'HATA'}")

    # 2. Tam Yığın Döngü Hızı
    print("\n [2] Tam Yığın RTOS Döngü Performansı...")
    print(f"     -> Ortalama Döngü Süresi   : {metrikler['cycle_ortalama_us']:.3f} µs (P99: {metrikler['cycle_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik FPS Kapasitesi: {metrikler['saniyelik_fps_kapasitesi']:,} FPS")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Faz 7 Master Capstone Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaCapstoneGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_faz7_capstone_infotainment_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 FAZ 7 BÜYÜK CAPSTONE BAŞARIYLA TAMAMLANDI! FAZ 7 %100 BİTTİ! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
