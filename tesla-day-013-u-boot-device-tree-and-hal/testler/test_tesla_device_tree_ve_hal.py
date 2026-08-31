"""
Tesla U-Boot, Device Tree ve HAL Birim Testleri (PyTest)
========================================================
Bu test paketi; Device Tree (.dts) dugum yuklemesini, U-Boot acilis
surelerini ve I2C/SPI HAL sensor okuma fonksiyonlarini dogrular.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import pytest
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_device_tree_ve_hal import (
    TeslaDeviceTreeParser,
    TeslaUBootAcilisSekansi,
    TeslaDonanimSoyutlamaKatmani
)


def test_device_tree_dugum_ayristirma():
    """DTS ağacına eklenen donanım düğümlerinin compatible dizesiyle bulunduğu test edilir."""
    dt = TeslaDeviceTreeParser()
    dt.standart_tesla_hw4_agacini_yukle()

    dugum_i2c = dt.dugum_bul_compatible("ti,tmp102")
    assert dugum_i2c is not None
    assert dugum_i2c.reg_adresi == 0x48

    dugum_spi = dt.dugum_bul_compatible("invensense,icm42688")
    assert dugum_spi is not None
    assert dugum_spi.saat_hizi_hz == 10000000


def test_uboot_acilis_asamalari():
    """U-Boot ve Kernel açılışının 500 ms altında (otomotiv fast-boot standardı) tamamlandığı test edilir."""
    uboot = TeslaUBootAcilisSekansi()
    sonuclar = uboot.acilisi_gerceklestir()

    assert sonuclar["hizli_acilis_basarili_mi"] is True
    assert sonuclar["toplam_acilis_suresi_ms"] < 500.0
    assert "ROM_Bootloader" in sonuclar["asamalar"]


def test_i2c_sicaklik_sensor_hal():
    """HAL katmanından I2C sıcaklık sensörlerinin doğru okunduğu test edilir."""
    dt = TeslaDeviceTreeParser()
    dt.standart_tesla_hw4_agacini_yukle()
    hal = TeslaDonanimSoyutlamaKatmani(dt)

    t_inlet = hal.i2c_sicaklik_oku(0x48)
    t_outlet = hal.i2c_sicaklik_oku(0x49)

    assert t_inlet == 32.5
    assert t_outlet == 38.2
    assert hal.i2c_sicaklik_oku(0x99) == -273.15  # Geçersiz adres


def test_spi_imu_sensor_hal():
    """HAL katmanından SPI 6-eksenli IMU verilerinin okunduğu test edilir."""
    dt = TeslaDeviceTreeParser()
    dt.standart_tesla_hw4_agacini_yukle()
    hal = TeslaDonanimSoyutlamaKatmani(dt)

    imu = hal.spi_imu_oku()
    assert "ivme_z_g" in imu
    assert imu["ivme_z_g"] == 1.00  # 1G yerçekimi
    assert "cayro_z_dps" in imu
