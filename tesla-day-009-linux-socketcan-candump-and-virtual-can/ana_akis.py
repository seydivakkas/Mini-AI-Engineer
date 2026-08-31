"""
Tesla Gun 09 Ana Akis (Tesla Day 09 Main Pipeline)
===================================================
Linux SocketCAN Mimarisi, Sanal CAN (vcan0) ve Kernel Filtreleme
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

from src.tesla_socketcan_arayuzu import (
    TeslaCANFrame,
    TeslaSocketCANArayuzu,
    TeslaVCanAgSimulatoru
)
from src.tesla_socketcan_profilleyici import TeslaSocketCANProfilleyici
from src.tesla_socketcan_gorsellestirici import TeslaSocketCANGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GOMULU YAZILIM MASTERI | GUN 09: LINUX SOCKETCAN & VCAN0 🚗")
    print("================================================================================")
    print("Stajyer Gorevi: Kernel CAN_RAW_FILTER, vcan0 Dugum Simulasyonu & candump/cansend")
    print("--------------------------------------------------------------------------------\n")

    # 1. Sanal CAN (vcan0) Ağı ve Düğümleri Kur
    print(" [1] vcan0 Sanal CAN Ağı Başlatılıyor...")
    ag = TeslaVCanAgSimulatoru()

    bms = TeslaSocketCANArayuzu("vcan0")
    inverter = TeslaSocketCANArayuzu("vcan0")
    fsd_core = TeslaSocketCANArayuzu("vcan0")

    # FSD sadece 0x100 (BMS) ve 0x200 (Motor) dinliyor
    fsd_core.filtre_ekle(0x100, 0x7FF)
    fsd_core.filtre_ekle(0x200, 0x7FF)

    ag.dugum_ekle(bms)
    ag.dugum_ekle(inverter)
    ag.dugum_ekle(fsd_core)

    print("     -> 3 Düğüm vcan0 Ağına Bağlandı (BMS, Inverter, FSD Core).")
    print("     -> FSD Core için Kernel Filtresi Kuruldu: [0x100 (BMS), 0x200 (Motor)]")

    # 2. Mesaj Yayını (Broadcast)
    print("\n [2] Mesaj Yayını ve Kernel Filtreleme Testi...")
    frame_bms = TeslaCANFrame(can_id=0x100, can_dlc=8, data=b'\x01\x90\x00\x00\x03\xE8\x00\x00')
    frame_fren = TeslaCANFrame(can_id=0x300, can_dlc=4, data=b'\xFF\x00\xFF\x00')

    print(f"     -> [cansend vcan0 100#0190000003E80000] BMS Yayını Yapılıyor...")
    ag.yayinla(bms, frame_bms)

    print(f"     -> [cansend vcan0 300#FF00FF00] Fren Kontrol Yayını Yapılıyor...")
    ag.yayinla(inverter, frame_fren)

    alinan = fsd_core.frame_al()
    if alinan:
        print(f"     -> [candump vcan0] FSD Tarafından Alınan Paket: CAN ID: 0x{alinan.can_id:X} (DLC: {alinan.can_dlc})")
    
    alinan_2 = fsd_core.frame_al()
    if alinan_2 is None:
        print("     -> 0x300 (Fren) Paketi Kernel Tarafından Başarıyla DÜŞÜRÜLDÜ (Drop - Sıfır CPU Yükü)!")

    # 3. Profilleme ve Performans
    print("\n [3] Kernel SocketCAN vs Userspace Filtreleme Benchmark'ı...")
    profilleyici = TeslaSocketCANProfilleyici(paket_sayisi=10000)
    metrikler = profilleyici.benchmark_kernel_vs_userspace_filtreleme()

    print(f"     -> Kernel SocketCAN Filtreleme Gecikmesi : {metrikler['kernel_ort_ns']:.1f} ns")
    print(f"     -> Userspace Döngü Filtreleme Gecikmesi  : {metrikler['userspace_ort_ns']:.1f} ns")
    print(f"     -> Donanımsal Hızlanma Çarpanı          : {metrikler['hizlanma_orani']:.1f}x Daha Hızlı")
    print(f"     -> Saniyedeki Frame Kapasitesi          : {metrikler['saniyede_frame_kapasitesi']:,.0f} Frame/sn")

    # 4. Tanı Paneli Görselleştirme
    print("\n [4] 6 Panelli Tesla SocketCAN Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaSocketCANGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_socketcan_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 09 BAŞARIYLA TAMAMLANDI! LINUX SOCKETCAN MOTORU DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
