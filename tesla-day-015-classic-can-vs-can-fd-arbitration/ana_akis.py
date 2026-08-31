"""
Tesla Gun 15 Ana Akis (Tesla Day 15 Main Pipeline)
===================================================
Klasik CAN vs CAN-FD & Wired-AND Donanimsal Arbitrasyon
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

from src.tesla_can_fd_protokolu import (
    TeslaKlasikCANFrame,
    TeslaCANFDFrame,
    TeslaCANArbitrasyonSimulasyonu
)
from src.tesla_can_fd_profilleyici import TeslaCANFDProfilleyici
from src.tesla_can_fd_gorsellestirici import TeslaCANFDGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GOMULU YAZILIM MASTERI | GUN 15: KLASIK CAN VS CAN-FD 🚗")
    print("================================================================================")
    print("Stajyer Gorevi: 64-Byte Payload, 5 Mbps BRS & Wired-AND Donanımsal Arbitrasyon")
    print("--------------------------------------------------------------------------------\n")

    # 1. Çerçeve Boyut ve İletim Süresi Karşılaştırması
    print(" [1] CAN 2.0B vs CAN-FD Çerçeveleri Oluşturuluyor...")
    f_klasik = TeslaKlasikCANFrame(can_id=0x100, veri=b'BMS_STAT')
    f_fd = TeslaCANFDFrame(can_id=0x100, veri=b'A' * 64, brs_aktif_mi=True)

    print(f"     -> Klasik CAN (8-byte)  : İletim Süresi = {f_klasik.iletim_suresi_us_hesapla():.1f} µs (DLC: {f_klasik.dlc})")
    print(f"     -> CAN-FD (64-byte, BRS): İletim Süresi = {f_fd.iletim_suresi_us_hesapla():.1f} µs (DLC: {f_fd.dlc}, 8x Daha Fazla Veri!)")

    # 2. Wired-AND Arbitrasyon (Tahkimat) Testi
    print("\n [2] 3 Farklı Araç Modülü Aynı Anda Hatta Mesaj Gönderiyor (Arbitrasyon Yarışı)...")
    sim = TeslaCANArbitrasyonSimulasyonu()
    sim.mesaj_ekle("Tesla_Infotainment", 0x380, b'SPOTIFY_TRACK', "Medya Bilgi Ekranı (0x380)")
    sim.mesaj_ekle("Tesla_Fren_Modulu", 0x010, b'AEB_ACTIVE', "ASIL-D Acil Fren (0x010)")
    sim.mesaj_ekle("Tesla_Surucu_Motor", 0x120, b'TORQUE_500NM', "Sürücü Motoru (0x120)")

    arb = sim.arbitrasyon_yaristir()
    kazanan = arb["kazanan"]
    print(f"     -> 🏆 ARBİTRASYONU KAZANAN : {kazanan['dugum_adi']} (CAN ID: {hex(kazanan['can_id'])}) - {kazanan['aciklama']}")
    print("     -> ❌ ELENEN DÜĞÜMLER:")
    for e in arb["elenenler"]:
        print(f"        * {e['dugum_adi']:<20} (CAN ID: {hex(e['can_id'])}) -> {e['sebep']}")

    # 3. Performans ve Bant Genişliği Benchmark'ı
    print("\n [3] Protokol Bant Genişliği ve Hız Analizi...")
    profilleyici = TeslaCANFDProfilleyici()
    metrikler = profilleyici.benchmark_can_vs_can_fd()

    print(f"     -> Klasik CAN Efektif Bant Genişliği: {metrikler['klasik_bant_kbps']:.1f} kbps")
    print(f"     -> CAN-FD Efektif Bant Genişliği    : {metrikler['can_fd_bant_kbps']:.1f} kbps")
    print(f"     -> Bant Genişliği Artış Çarpanı     : {metrikler['bant_genisligi_carpani']:.1f}x Daha Yüksek")

    # 4. Tanı Paneli Görselleştirme
    print("\n [4] 6 Panelli Tesla CAN-FD Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaCANFDGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_can_fd_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 15 BAŞARIYLA TAMAMLANDI! CAN-FD VE ARBİTRASYON DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
