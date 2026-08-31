"""
Tesla FreeRTOS Çekirdek Birim Testleri (PyTest)
================================================
Bu test paketi; Preemptive görev çizelgelemesini, kuyruk senkronizasyonunu,
Mutex kilit mekanizmasını ve Öncelik Mirası (Priority Inheritance) protokolünü doğrular.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_freertos_cekirdek import (
    FreeRTOSScheduler,
    FreeRTOSQueue,
    FreeRTOSMutex,
    FreeRTOSTask,
    TaskState
)


def test_freertos_task_olusturma_ve_oncelik():
    """Görevlerin doğru öncelikle oluşturulduğu ve hazır duruma geçtiği test edilir."""
    sched = FreeRTOSScheduler()
    t1 = sched.create_task("TaskA", priority=3)
    t2 = sched.create_task("TaskB", priority=7)

    assert len(sched.tasks) == 2
    assert t2 > t1  # Yüksek öncelik sıralamada üstündür
    assert t1.state == TaskState.READY


def test_freertos_preemptive_cizelgeleme():
    """Çizelgeleyicinin her zaman en yüksek öncelikli hazır görevi çalıştırdığı test edilir."""
    sched = FreeRTOSScheduler()
    t_low = sched.create_task("LowTask", priority=2)
    t_high = sched.create_task("HighTask", priority=10)

    # 1. Adım: HighTask icra edilmeli
    active = sched.step_tick()
    assert active == t_high
    assert t_high.state == TaskState.RUNNING

    # HighTask engellensin (delay)
    sched.delay_current_task(ticks_to_delay=5)
    assert t_high.state == TaskState.BLOCKED

    # 2. Adım: HighTask engelli olduğundan LowTask icra edilmeli
    active2 = sched.step_tick()
    assert active2 == t_low
    assert t_low.state == TaskState.RUNNING


def test_freertos_kuyruk_operasyonlari():
    """Thread-safe kuyruk gönderme, alma ve taşma/boşluk davranışları test edilir."""
    q = FreeRTOSQueue(length=2)

    assert q.is_empty() is True
    assert q.is_full() is False

    # 2 eleman ekle
    assert q.send("Veri1") is True
    assert q.send("Veri2") is True
    assert q.is_full() is True

    # 3. eleman taşmalı
    assert q.send("Veri3") is False

    # Elemanları çek
    assert q.receive() == "Veri1"
    assert q.receive() == "Veri2"
    assert q.is_empty() is True
    assert q.receive() is None


def test_freertos_priority_inheritance():
    """Öncelik Mirası (Priority Inheritance) ile Mars Pathfinder tipi kilitlenmelerin önlendiği test edilir."""
    sched = FreeRTOSScheduler()
    mutex = FreeRTOSMutex(name="SPI_Bus_Mutex")

    t_low = sched.create_task("LowTask", priority=1)
    t_mid = sched.create_task("MidTask", priority=5)
    t_high = sched.create_task("HighTask", priority=10)

    # 1. LowTask Mutex'i alır
    assert mutex.take(t_low) is True
    assert mutex.holder_task == t_low

    # 2. HighTask Mutex'i ister -> Mutex dolu olduğu için HighTask BLOCKED olur
    #    ve LowTask'in dinamik önceliği 10'a yükselir!
    assert mutex.take(t_high) is False
    assert t_high.state == TaskState.BLOCKED
    assert t_low.priority == 10  # Öncelik Mirası gerçekleşti!

    # 3. MidTask (Öncelik 5), LowTask'i (Öncelik 10) bölemez (Preempt edemez)!
    # 4. LowTask Mutex'i bırakır
    assert mutex.give(t_low) is True
    # LowTask orijinal önceliğine geri döner
    assert t_low.priority == 1
    # Mutex artık HighTask'e geçmiştir ve HighTask READY olmuştur
    assert mutex.holder_task == t_high
    assert t_high.state == TaskState.READY
