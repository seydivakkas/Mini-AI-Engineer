"""
Tesla Gun 13 Ana Akis (Tesla Day 13 Main Pipeline)
===================================================
U-Boot Bootloader, Device Tree (.dts) ve Donanim Soyutlama Katmani (HAL)
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

from src.tesla_device_tree_ve_hal import (
    TeslaDeviceTreeParser,
    TeslaUBootAcilisSekansi,
    TeslaDonanimSoyutlamaKatmani
)
from src.tesla_hal_profilleyici import TeslaHALProfilleyici
from src.tesla_hal_gorsellestirici import TeslaHALGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GOMULU YAZILIM MASTERI | GUN 13: U-BOOT, DEVICE TREE & HAL 🚗")
    print("================================================================================")
    print("Stajyer Gorevi: DTS Agaci Yukleme, Hizli Acilis (<500ms) & I2C/SPI HAL Entegrasyonu")
    print("--------------------------------------------------------------------------------\n")

    # 1. Device Tree (.dts) Yükleme
    print(" [1] Tesla HW4 SoC Device Tree (.dts) Düğümleri Ayrıştırılıyor...")
    dt = TeslaDeviceTreeParser()
    dt.standart_tesla_hw4_agacini_yukle()
    print(f"     -> Toplam {len(dt.dugumler)} Donanım Düğümü Yüklendi (I2C, TMP102, SPI, ICM42688).")

    # 2. U-Boot Fast Boot Aşamaları
    print("\n [2] U-Boot Fast Boot ve Kernel Açılış Sekansı...")
    uboot = TeslaUBootAcilisSekansi()
    boot_sonuclari = uboot.acilisi_gerceklestir()

    for asama, sure in boot_sonuclari["asamalar"].items():
        print(f"     -> {asama:<25}: {sure:.1f} ms")
    print(f"     -> TOPLAM AÇILIŞ SÜRESİ    : {boot_sonuclari['toplam_acilis_suresi_ms']:.1f} ms (Hedef < 500 ms: {'GÜVENLİ' if boot_sonuclari['hizli_acilis_basarili_mi'] else 'YAVAŞ'})")

    # 3. C++ Donanım Soyutlama Katmanı (HAL) Okumaları
    print("\n [3] HAL Katmanı Üzerinden Sensör Okumaları Yapılıyor...")
    hal = TeslaDonanimSoyutlamaKatmani(dt)
    t_in = hal.i2c_sicaklik_oku(0x48)
    t_out = hal.i2c_sicaklik_oku(0x49)
    imu = hal.spi_imu_oku()

    print(f"     -> [I2C TMP102] Batarya Giriş Sıcaklığı: {t_in:.1f} °C | Çıkış: {t_out:.1f} °C")
    print(f"     -> [SPI ICM-42688] IMU Yerçekimi (Z-Axis): {imu['ivme_z_g']:.2f} G | Yaw Oranı: {imu['cayro_z_dps']:.2f} dps")

    # 4. Profilleme ve Karşılaştırma
    print("\n [4] Device Tree HAL vs Dinamik Aygıt Tarama Benchmark'ı...")
    profilleyici = TeslaHALProfilleyici(ornek_sayisi=5000)
    metrikler = profilleyici.benchmark_hal_vs_dinamik_tarama()

    print(f"     -> Device Tree HAL Erişim Gecikmesi     : {metrikler['hal_ortalama_us']:.3f} µs")
    print(f"     -> Dinamik Runtime Aygıt Tarama Süresi  : {metrikler['tarama_ortalama_us']:.3f} µs")
    print(f"     -> Hızlanma Çarpanı                     : {metrikler['hizlanma_orani']:.1f}x Daha Hızlı")

    # 5. Tanı Paneli Görselleştirme
    print("\n [5] 6 Panelli Tesla HAL Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaHALGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_hal_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 13 BAŞARIYLA TAMAMLANDI! U-BOOT & DEVICE TREE HAL DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
