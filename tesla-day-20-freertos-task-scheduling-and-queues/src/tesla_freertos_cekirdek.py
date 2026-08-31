"""
Tesla FreeRTOS Çekirdek, Görev Senkronizasyonu ve Kuyruk Modülü
===============================================================
Bu modül; FreeRTOS standartlarında Preemptive Görev Çizelgeleme,
Görev Kontrol Bloğu (TCB), Thread-Safe Kuyruklar (Queue), İkili Semaforlar
ve Öncelik Tersine Çevrilmesini (Priority Inversion) engelleyen
Öncelik Mirası (Priority Inheritance) Mutex mekanizmalarını gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
import heapq
import time
import collections


class TaskState(IntEnum):
    READY = 0
    RUNNING = 1
    BLOCKED = 2
    SUSPENDED = 3


@dataclass
class FreeRTOSTask:
    name: str
    priority: int                   # Dinamik öncelik (0: En düşük, Örn: 31: En yüksek)
    base_priority: int              # Orijinal temel öncelik
    stack_size_words: int = 128
    state: TaskState = TaskState.READY
    runtime_ticks: int = 0
    blocked_until_tick: int = 0
    waiting_on_object: Optional[Any] = None

    def __lt__(self, other: 'FreeRTOSTask') -> bool:
        return self.priority < other.priority

    def __gt__(self, other: 'FreeRTOSTask') -> bool:
        return self.priority > other.priority


class FreeRTOSQueue:
    """
    FreeRTOS Thread-Safe ve Deterministik Kuyruk Gerçeklemesi.
    (xQueueCreate, xQueueSend, xQueueReceive)
    """
    def __init__(self, length: int, item_name: str = "Generic"):
        self.length = length
        self.item_name = item_name
        self.queue: collections.deque = collections.deque(maxlen=length)
        self.waiting_send_tasks: List[FreeRTOSTask] = []
        self.waiting_recv_tasks: List[FreeRTOSTask] = []

    def is_full(self) -> bool:
        return len(self.queue) >= self.length

    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def send(self, item: Any, task: Optional[FreeRTOSTask] = None) -> bool:
        if self.is_full():
            if task:
                task.state = TaskState.BLOCKED
                task.waiting_on_object = self
                self.waiting_send_tasks.append(task)
            return False
        self.queue.append(item)
        # Bekleyen alıcı görev varsa hazır yap
        if self.waiting_recv_tasks:
            recv_task = self.waiting_recv_tasks.pop(0)
            recv_task.state = TaskState.READY
            recv_task.waiting_on_object = None
        return True

    def receive(self, task: Optional[FreeRTOSTask] = None) -> Optional[Any]:
        if self.is_empty():
            if task:
                task.state = TaskState.BLOCKED
                task.waiting_on_object = self
                self.waiting_recv_tasks.append(task)
            return None
        item = self.queue.popleft()
        # Bekleyen gönderici görev varsa hazır yap
        if self.waiting_send_tasks:
            send_task = self.waiting_send_tasks.pop(0)
            send_task.state = TaskState.READY
            send_task.waiting_on_object = None
        return item


class FreeRTOSMutex:
    """
    Öncelik Mirası (Priority Inheritance) Destekli FreeRTOS Mutex.
    (xSemaphoreCreateMutex, xSemaphoreTake, xSemaphoreGive)
    """
    def __init__(self, name: str = "TeslaBMSMutex"):
        self.name = name
        self.holder_task: Optional[FreeRTOSTask] = None
        self.waiting_tasks: List[FreeRTOSTask] = []

    def take(self, task: FreeRTOSTask) -> bool:
        if self.holder_task is None:
            self.holder_task = task
            return True

        if self.holder_task == task:
            return True  # Recursive take

        # Mutex başka bir görevde; Öncelik Mirasını (Priority Inheritance) uygula!
        if task.priority > self.holder_task.priority:
            # Düşük öncelikli tutucu görev, yüksek öncelikli bekleyen görevin önceliğini devralır
            self.holder_task.priority = task.priority

        task.state = TaskState.BLOCKED
        task.waiting_on_object = self
        self.waiting_tasks.append(task)
        return False

    def give(self, task: FreeRTOSTask) -> bool:
        if self.holder_task != task:
            return False  # Sadece tutan görev bırakabilir

        # Önceliği orijinal temel önceliğe geri döndür
        task.priority = task.base_priority
        self.holder_task = None

        if self.waiting_tasks:
            # Bekleyenler arasından en yüksek öncelikliyi seç
            self.waiting_tasks.sort(key=lambda t: t.priority, reverse=True)
            next_task = self.waiting_tasks.pop(0)
            self.holder_task = next_task
            next_task.state = TaskState.READY
            next_task.waiting_on_object = None
        return True


class FreeRTOSScheduler:
    """
    FreeRTOS Deterministik Çekirdek Çizelgeleyicisi (Preemptive Scheduler).
    1 kHz (1 ms) zamanlayıcı kesmesiyle (SysTick) görevleri koordine eder.
    """
    def __init__(self, max_priorities: int = 32):
        self.max_priorities = max_priorities
        self.tasks: List[FreeRTOSTask] = []
        self.current_task: Optional[FreeRTOSTask] = None
        self.tick_count = 0
        self.context_switch_count = 0
        self.execution_log: List[Tuple[int, str, int]] = []  # (Tick, TaskName, Priority)

    def create_task(self, name: str, priority: int, stack_size: int = 128) -> FreeRTOSTask:
        task = FreeRTOSTask(name=name, priority=priority, base_priority=priority, stack_size_words=stack_size)
        self.tasks.append(task)
        return task

    def step_tick(self) -> Optional[FreeRTOSTask]:
        """Bir SysTick (1 ms) adımı ilerletir ve en yüksek öncelikli hazır görevi çalıştırır."""
        self.tick_count += 1

        # 1. Engellenmiş görevlerin sürelerini güncelle
        for t in self.tasks:
            if t.state == TaskState.BLOCKED and t.blocked_until_tick > 0:
                if self.tick_count >= t.blocked_until_tick:
                    t.state = TaskState.READY
                    t.blocked_until_tick = 0

        # 2. Hazır görevleri filtrele ve öncelik sırasına koy
        ready_tasks = [t for t in self.tasks if t.state in (TaskState.READY, TaskState.RUNNING)]
        if not ready_tasks:
            self.current_task = None
            return None

        # En yüksek öncelikli görevi seç
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)
        highest_task = ready_tasks[0]

        # Context Switch Kontrolü
        if self.current_task != highest_task:
            if self.current_task and self.current_task.state == TaskState.RUNNING:
                self.current_task.state = TaskState.READY
            self.context_switch_count += 1
            self.current_task = highest_task
            highest_task.state = TaskState.RUNNING

        highest_task.runtime_ticks += 1
        self.execution_log.append((self.tick_count, highest_task.name, highest_task.priority))
        return highest_task

    def delay_current_task(self, ticks_to_delay: int):
        """vTaskDelay eşdeğeri: Mevcut görevi belirli tick kadar uyutur."""
        if self.current_task:
            self.current_task.state = TaskState.BLOCKED
            self.current_task.blocked_until_tick = self.tick_count + ticks_to_delay
