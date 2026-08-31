"""
Tesla Gün 28 Ana Akış (Tesla Day 28 Main Pipeline)
===================================================
İnvertör ve PMSM Motor Kontrolü: Field Oriented Control (FOC)
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

from src.tesla_foc_motor_kontrolcusu import (
    TeslaMotorParameters,
    TeslaFOCController,
    ClarkeTransform,
    ParkTransform
)
from src.tesla_foc_profilleyici import TeslaFOCProfilleyici
from src.tesla_foc_gorsellestirici import TeslaFOCGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GÖMÜLÜ YAZILIM MASTERI | GÜN 28: FOC MOTOR KONTROLCÜSÜ 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Clarke/Park Dönüşümleri, dq Akı/Tork Ayrıştırması & 10 kHz FOC")
    print("--------------------------------------------------------------------------------\n")

    # 1. 350 Nm Tam Gaz İvmelenme Simülasyonu
    print(" [1] 3000 RPM Hızda 0'dan 350 Nm Ludicrous Tork Basamak Testi Başlatılıyor...")
    profilleyici = TeslaFOCProfilleyici(sim_adimlari=1000)
    metrikler = profilleyici.benchmark_foc_dongusu()

    print(f"     -> Hedeflenen Maksimum Tork    : 350.0 Nm")
    print(f"     -> Üretilen Elektromanyetik Tork: {metrikler['max_torque_nm']:.1f} Nm")
    print(f"     -> Pik Faz Akımı (i_a)         : {max(metrikler['i_a']):.1f} A")
    print(f"     -> Kuadratür Akımı (i_q)       : {max(metrikler['i_q']):.1f} A (Tork Üreten)")
    print(f"     -> Doğrudan Akı Akımı (i_d)    : {metrikler['i_d'][-1]:.2f} A (Manyetik Akı = 0)")

    # 2. 10 kHz RTOS Akım Çevrim Gecikmesi
    print("\n [2] 10 kHz (100 µs Bütçeli) FOC Akım Döngüsü RTOS Performansı...")
    print(f"     -> Ortalama FOC Adım Süresi    : {metrikler['foc_step_ortalama_us']:.3f} µs (P99: {metrikler['foc_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik FOC Kapasitesi    : {metrikler['saniyelik_foc_adimi']:,} Adım/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FOC Motor Kontrol Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaFOCGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_foc_motor_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 28 BAŞARIYLA TAMAMLANDI! FOC İNVERTÖR & PMSM MOTOR KONTROLÜ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
