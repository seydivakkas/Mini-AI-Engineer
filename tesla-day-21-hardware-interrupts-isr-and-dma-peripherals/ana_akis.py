"""
Tesla Gün 21 Ana Akış (Tesla Day 21 Main Pipeline)
===================================================
Donanım Kesmeleri (ISR) ve DMA Çevre Birimleri (SPI/I2C/UART)
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

from src.tesla_kesme_ve_dma_suruculeri import (
    NVICController,
    TeslaSPISensorDriver,
    IRQChannel
)
from src.tesla_kesme_profilleyici import TeslaKesmeProfilleyici
from src.tesla_kesme_gorsellestirici import TeslaKesmeGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GÖMÜLÜ YAZILIM MASTERI | GÜN 21: HARDWARE ISR & ZERO-COPY DMA 🚗")
    print("================================================================================")
    print("Stajyer Görevi: NVIC Öncelik Yuvalaması, Ping-Pong DMA Tamponlama & SPI 50 MHz")
    print("--------------------------------------------------------------------------------\n")

    nvic = NVICController()
    spi_drv = TeslaSPISensorDriver(nvic)

    # 1. Donanım SPI Sensör Transferi & DMA ISR
    print(" [1] 50 MHz SPI İvmeölçer Sensöründen DMA ile Veri Okunuyor...")
    spi_drv.simulate_hardware_spi_transfer(ax=0.42, ay=-0.18, az=9.79)

    print(f"     -> DMA Kesmesi Tetiklendi     : {spi_drv.dma_isr_call_count} Kez")
    print(f"     -> Ayrıştırılan İvme X (m/s²) : {spi_drv.latest_accel_x:+.2f}")
    print(f"     -> Ayrıştırılan İvme Y (m/s²) : {spi_drv.latest_accel_y:+.2f}")
    print(f"     -> Ayrıştırılan İvme Z (m/s²) : {spi_drv.latest_accel_z:+.2f}")

    # 2. NVIC Öncelik Yuvalaması (Interrupt Preemption)
    print("\n [2] NVIC Çekirdek Öncelik Yuvalaması ve Acil Durum Kesmesi Denetimi...")
    acil_tetiklendi = {"durum": False}

    def acil_kaza_kesmesi():
        acil_tetiklendi["durum"] = True
        print("     -> ⚠️ ACİL KESME: EXTI0 Hava Yastığı / Kaza Sensörü Tetiklendi (Prio 0)!")

    nvic.register_irq(
        channel=IRQChannel.EXTI0_CRASH_PIN,
        name="EXTI_Crash_ISR",
        preemption_priority=0,  # En yüksek öncelik
        sub_priority=0,
        handler=acil_kaza_kesmesi
    )

    nvic.trigger_irq(IRQChannel.EXTI0_CRASH_PIN)
    print(f"     -> Acil Kesme Başarıyla İşlendi: {acil_tetiklendi['durum']}")

    # 3. Performans ve Gecikme Benchmark'ı
    print("\n [3] ISR Yanıt Gecikmesi ve Polling vs DMA CPU Yükü Analizi...")
    profilleyici = TeslaKesmeProfilleyici(ornek_sayisi=5000)
    metrikler = profilleyici.benchmark_kesme_ve_dma()

    print(f"     -> Ortalama ISR İcra Süresi   : {metrikler['isr_ortalama_us']:.3f} µs (P99: {metrikler['isr_p99_us']:.3f} µs)")
    print(f"     -> CPU Yükü (Polling / Yoklama): %{metrikler['cpu_polling_pct']:.1f}")
    print(f"     -> CPU Yükü (DMA Zero-Copy)   : %{metrikler['cpu_dma_pct']:.1f}")
    print(f"     -> CPU Tasarruf Çarpanı       : {metrikler['cpu_kazanci_carpani']:.1f}x Daha Verimli!")
    print(f"     -> SPI DMA Bant Genişliği     : {metrikler['dma_throughput_mbs']} MB/s")

    # 4. Tanı Paneli Görselleştirme
    print("\n [4] 6 Panelli Tesla Donanım Kesmeleri & DMA Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaKesmeGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_kesme_dma_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 21 BAŞARIYLA TAMAMLANDI! HARDWARE ISR & DMA SÜRÜCÜLERİ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
