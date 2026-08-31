"""
Tesla Donanım Kesmeleri (ISR) ve DMA Sürücüleri Modülü
======================================================
Bu modül; ARM Cortex-M / TriCore mimarilerindeki NVIC (Nested Vectored
Interrupt Controller), DMA (Direct Memory Access) Çift Tamponlu (Ping-Pong)
Aktarım ve SPI/I2C/UART donanım çevre birimi sürücülerini gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
import struct
import time
import numpy as np


class IRQChannel(IntEnum):
    DMA1_STREAM0_SPI_RX = 11
    DMA1_STREAM1_SPI_TX = 12
    USART1_GLOBAL_IRQ   = 37
    I2C1_EV_IRQ         = 31
    I2C1_ER_IRQ         = 32
    EXTI0_CRASH_PIN     = 6


@dataclass
class IRQVector:
    channel: IRQChannel
    name: str
    preemption_priority: int  # 0: En yüksek öncelik
    sub_priority: int
    isr_handler: Callable[[], None]
    enabled: bool = True
    pending: bool = False
    active: bool = False


class NVICController:
    """
    Nested Vectored Interrupt Controller (NVIC) Simülatörü.
    Öncelik yuvalaması (Preemption / Nesting) ve Kesme Kuyruk Yönetimi.
    """
    def __init__(self):
        self.vectors: Dict[IRQChannel, IRQVector] = {}
        self.active_irq_stack: List[IRQVector] = []
        self.total_interrupts_handled = 0

    def register_irq(self, channel: IRQChannel, name: str, preemption_priority: int, sub_priority: int, handler: Callable[[], None]):
        self.vectors[channel] = IRQVector(
            channel=channel,
            name=name,
            preemption_priority=preemption_priority,
            sub_priority=sub_priority,
            isr_handler=handler
        )

    def trigger_irq(self, channel: IRQChannel) -> bool:
        """Donanımsal bir olay sonucu kesme tetikler."""
        if channel not in self.vectors:
            return False
        vec = self.vectors[channel]
        if not vec.enabled:
            return False

        # Mevcut aktif kesmeden daha yüksek öncelikli mi kontrol et (Öncelik yuvalaması)
        if self.active_irq_stack:
            current_active = self.active_irq_stack[-1]
            if vec.preemption_priority < current_active.preemption_priority:
                # Yuvalama (Nesting): Yeni kesme mevcut kesmeyi böler!
                self._execute_isr(vec)
                return True
            else:
                vec.pending = True
                return False
        else:
            self._execute_isr(vec)
            return True

    def _execute_isr(self, vec: IRQVector):
        vec.active = True
        vec.pending = False
        self.active_irq_stack.append(vec)
        self.total_interrupts_handled += 1

        # ISR Çalıştır (Kritik kural: ISR içinde bloklama yapılmaz!)
        vec.isr_handler()

        self.active_irq_stack.pop()
        vec.active = False


class DMABufferMode(IntEnum):
    NORMAL = 0
    CIRCULAR = 1
    DOUBLE_BUFFER_PING_PONG = 2


class DMACircularController:
    """
    DMA (Direct Memory Access) Çevrimsel & Ping-Pong Tampon Kontrolcüsü.
    CPU'yu meşgul etmeden çevre biriminden RAM'e veri pompalar.
    """
    def __init__(self, buffer_size: int = 1024, mode: DMABufferMode = DMABufferMode.DOUBLE_BUFFER_PING_PONG):
        self.buffer_size = buffer_size
        self.mode = mode
        self.memory_buffer_0 = bytearray(buffer_size // 2)
        self.memory_buffer_1 = bytearray(buffer_size // 2)
        self.current_target_is_buffer_1 = False
        self.transferred_bytes_total = 0
        
        # Kesme Bayrakları
        self.half_transfer_flag = False
        self.transfer_complete_flag = False

    def push_peripheral_data(self, data: bytes, nvic: Optional[NVICController] = None):
        """Çevre biriminden gelen baytları doğrudan RAM tamponuna yazar."""
        chunk_len = len(data)
        target = self.memory_buffer_1 if self.current_target_is_buffer_1 else self.memory_buffer_0
        
        # Tamponu doldur
        target[:chunk_len] = data[:len(target)]
        self.transferred_bytes_total += chunk_len

        # Yarım Tampon Dolumu (Half Transfer Complete)
        self.half_transfer_flag = True
        
        # Tam Tampon Dolumu (Transfer Complete) -> Tampon Takası (Ping-Pong Swap)
        self.transfer_complete_flag = True
        self.current_target_is_buffer_1 = not self.current_target_is_buffer_1

        # DMA Transfer Complete Kesmesini Tetikle
        if nvic:
            nvic.trigger_irq(IRQChannel.DMA1_STREAM0_SPI_RX)

    def get_ready_buffer(self) -> bytes:
        """CPU'nun işlemeye hazır olduğu pasif tamponu döner."""
        # Şu an DMA buffer_1'e yazıyorsa CPU buffer_0'ı güvenle okur
        return bytes(self.memory_buffer_0 if self.current_target_is_buffer_1 else self.memory_buffer_1)


class TeslaSPISensorDriver:
    """
    Tesla 50 MHz Yüksek Hızlı SPI İvmeölçer / Jiroskop Sensör Sürücüsü (DMA Destekli).
    """
    def __init__(self, nvic: NVICController):
        self.nvic = nvic
        self.dma = DMACircularController(buffer_size=128, mode=DMABufferMode.DOUBLE_BUFFER_PING_PONG)
        self.latest_accel_x = 0.0
        self.latest_accel_y = 0.0
        self.latest_accel_z = 0.0
        self.dma_isr_call_count = 0

        # NVIC'ye DMA RX Kesmesini Kaydet (Öncelik: 2)
        self.nvic.register_irq(
            channel=IRQChannel.DMA1_STREAM0_SPI_RX,
            name="SPI_DMA_RX_ISR",
            preemption_priority=2,
            sub_priority=0,
            handler=self.handle_dma_rx_isr
        )

    def handle_dma_rx_isr(self):
        """Donanımsal SPI DMA Tamamlanma Kesme Servis Rutini (ISR)."""
        self.dma_isr_call_count += 1
        raw_bytes = self.dma.get_ready_buffer()
        if len(raw_bytes) >= 12:
            ax, ay, az = struct.unpack(">fff", raw_bytes[:12])
            self.latest_accel_x = ax
            self.latest_accel_y = ay
            self.latest_accel_z = az

    def simulate_hardware_spi_transfer(self, ax: float, ay: float, az: float):
        """Sensörün SPI hattından DMA tamponuna veri akıtması simülasyonu."""
        payload = struct.pack(">fff", ax, ay, az)
        self.dma.push_peripheral_data(payload, self.nvic)
