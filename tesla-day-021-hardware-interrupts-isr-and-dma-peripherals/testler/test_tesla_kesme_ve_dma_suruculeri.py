"""
Tesla Donanım Kesmeleri ve DMA Birim Testleri (PyTest)
======================================================
Bu test paketi; NVIC öncelik yuvalamasını, DMA Ping-Pong çift tampon
çalışmasını ve SPI sensör kesme işleyicilerini doğrular.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import struct
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_kesme_ve_dma_suruculeri import (
    NVICController,
    DMACircularController,
    TeslaSPISensorDriver,
    IRQChannel,
    DMABufferMode
)


def test_nvic_kesme_tetikleme_ve_kayit():
    """NVIC kesme kaydı ve temel tetikleme akışı test edilir."""
    nvic = NVICController()
    sayac = {"val": 0}

    def ornek_isr():
        sayac["val"] += 1

    nvic.register_irq(
        channel=IRQChannel.EXTI0_CRASH_PIN,
        name="Crash_ISR",
        preemption_priority=0,
        sub_priority=0,
        handler=ornek_isr
    )

    assert nvic.trigger_irq(IRQChannel.EXTI0_CRASH_PIN) is True
    assert sayac["val"] == 1
    assert nvic.total_interrupts_handled == 1


def test_nvic_oncelik_yuvalamasi_nesting():
    """Yüksek öncelikli kesmenin düşük öncelikli kesmeyi böldüğü (Nesting) test edilir."""
    nvic = NVICController()
    olay_sirasi = []

    def low_prio_isr():
        olay_sirasi.append("Low_Start")
        # Düşük öncelikli kesme içindeyken yüksek öncelikli kesme tetiklenir
        nvic.trigger_irq(IRQChannel.EXTI0_CRASH_PIN)
        olay_sirasi.append("Low_End")

    def high_prio_isr():
        olay_sirasi.append("High_Executed")

    nvic.register_irq(
        channel=IRQChannel.USART1_GLOBAL_IRQ,
        name="UART_ISR",
        preemption_priority=4,  # Düşük öncelik
        sub_priority=0,
        handler=low_prio_isr
    )
    nvic.register_irq(
        channel=IRQChannel.EXTI0_CRASH_PIN,
        name="Crash_ISR",
        preemption_priority=0,  # En yüksek öncelik
        sub_priority=0,
        handler=high_prio_isr
    )

    nvic.trigger_irq(IRQChannel.USART1_GLOBAL_IRQ)

    # Yuvalama sırası: Low_Start -> High_Executed -> Low_End
    assert olay_sirasi == ["Low_Start", "High_Executed", "Low_End"]


def test_dma_ping_pong_tampon_degisimi():
    """DMA tampon dolumu ve Ping-Pong çift tampon takası test edilir."""
    dma = DMACircularController(buffer_size=16, mode=DMABufferMode.DOUBLE_BUFFER_PING_PONG)
    
    # 8 bayt veri yaz (Buffer 0 dolmalı)
    dma.push_peripheral_data(b"12345678")
    assert dma.half_transfer_flag is True
    assert dma.transfer_complete_flag is True
    # Takas gerçekleşti, hazır tampon buffer 0 olmalı
    ready_data = dma.get_ready_buffer()
    assert ready_data[:8] == b"12345678"


def test_tesla_spi_sensor_surucusu():
    """SPI DMA transferi ve ivmeölçer verisinin ISR ile doğru ayrıştırıldığı test edilir."""
    nvic = NVICController()
    spi_drv = TeslaSPISensorDriver(nvic)

    spi_drv.simulate_hardware_spi_transfer(ax=1.25, ay=-0.45, az=9.80)

    assert spi_drv.dma_isr_call_count == 1
    assert pytest.approx(spi_drv.latest_accel_x, 0.01) == 1.25
    assert pytest.approx(spi_drv.latest_accel_y, 0.01) == -0.45
    assert pytest.approx(spi_drv.latest_accel_z, 0.01) == 9.80
