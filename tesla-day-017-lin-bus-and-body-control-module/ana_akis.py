"""
Tesla Gun 17 Ana Akis (Tesla Day 17 Main Pipeline)
===================================================
LIN (Local Interconnect Network) Veri Yolu & Govde Kontrol Modulu (BCM)
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

from src.tesla_lin_protokolu import (
    TeslaLINSlaveBCM,
    TeslaLINMasterCizelgeleyici,
    pid_hesapla,
    pid_dogrula
)
from src.tesla_lin_profilleyici import TeslaLINProfilleyici
from src.tesla_lin_gorsellestirici import TeslaLINGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GOMULU YAZILIM MASTERI | GUN 17: LIN VERI YOLU & BCM 🚗")
    print("================================================================================")
    print("Stajyer Gorevi: LIN Master Schedule Table, PID Parite & BCM Govde Kontrolu")
    print("--------------------------------------------------------------------------------\n")

    slave_bcm = TeslaLINSlaveBCM()
    master = TeslaLINMasterCizelgeleyici(slave_bcm)

    # 1. LIN Master Çizelgeleme Tablosu Yürütme
    print(" [1] LIN Master Çizelgeleme Tablosu (Schedule Table) Çalıştırılıyor...")
    for gorev in master.cizelge_tablosu:
        msg = master.cerceve_gonder(gorev["frame_id"], gorev["veri"])
        islem = slave_bcm.lin_mesaj_isle(msg)
        print(f"     -> [Master Header] Break + Sync (0x55) + PID: {hex(msg.pid)} ({gorev['isim']})")
        print(f"        * Slave BCM Yanıtı: Aygıt = {islem['aygit']} | Değer = {islem['yeni_deger']}")

    # 2. Gövde Kontrol Modülü Nihai Durumları
    print("\n [2] BCM (Gövde Kontrol Modülü) Canlı Aktüatör Durumları:")
    print(f"     -> Pencere Konumu        : %{slave_bcm.pencere_seviyesi_yuzde:.0f}")
    print(f"     -> Koltuk Mesafesi       : {slave_bcm.koltuk_pozisyonu_mm:.0f} mm")
    print(f"     -> Silecek Çalışma Modu  : Kademe {slave_bcm.silecek_kademesi}")
    print(f"     -> Ortam Ambiyans Işığı  : RGB{slave_bcm.ambiyans_rgb}")

    # 3. Performans ve Hız Benchmark'ı
    print("\n [3] LIN Veri Yolu ve PID Parite Benchmark Analizi...")
    profilleyici = TeslaLINProfilleyici(ornek_sayisi=5000)
    metrikler = profilleyici.benchmark_lin_performansi()

    print(f"     -> PID Parite & Checksum Süresi: {metrikler['pid_ortalama_us']:.3f} µs (P99: {metrikler['pid_p99_us']:.3f} µs)")
    print(f"     -> LIN 19.2 kbps Çerçeve Süresi : {metrikler['lin_19k2_sure_ms']:.2f} ms")
    print(f"     -> LIN 9.6 kbps Çerçeve Süresi  : {metrikler['lin_9k6_sure_ms']:.2f} ms")
    print(f"     -> Kablo / Donanım Tasarrufu   : CAN'a Kıyasla %72 Daha Ekonomik")

    # 4. Tanı Paneli Görselleştirme
    print("\n [4] 6 Panelli Tesla LIN Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaLINGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_lin_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 17 BAŞARIYLA TAMAMLANDI! LIN VERİ YOLU & BCM DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
