"""
Tesla FreeRTOS Profilleyici Modülü
===================================
Bu modül; FreeRTOS görev çizelgeleme gecikmesini, bağlam değiştirme
(Context Switch) süresini, kuyruk iletim verimini ve Öncelik Mirası
(Priority Inheritance) performansını ölçer.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_freertos_cekirdek import (
    FreeRTOSScheduler,
    FreeRTOSQueue,
    FreeRTOSMutex,
    TaskState
)


class TeslaFreeRTOSProfilleyici:
    """
    FreeRTOS Çekirdek Performans Profilleyicisi.
    """
    def __init__(self, ornek_sayisi: int = 5000):
        self.ornek_sayisi = ornek_sayisi

    def benchmark_freertos(self) -> Dict[str, Any]:
        # 1. Kuyruk (Queue) İletim Hızı ve Gecikmesi
        queue = FreeRTOSQueue(length=64)
        gecikmeler_kuyruk_us: List[float] = []
        for i in range(self.ornek_sayisi):
            t0 = time.perf_counter_ns()
            queue.send({"sensor_id": 0x12, "voltage": 3.85, "idx": i})
            _ = queue.receive()
            t1 = time.perf_counter_ns()
            gecikmeler_kuyruk_us.append(float(t1 - t0) / 1000.0)

        # 2. Öncelik Mirası (Priority Inheritance) Doğrulama Simülasyonu
        sched = FreeRTOSScheduler()
        mutex = FreeRTOSMutex(name="SharedBMSBus")

        t_low = sched.create_task(name="TelemetryLogger", priority=1)
        t_mid = sched.create_task(name="CabinFanControl", priority=5)
        t_high = sched.create_task(name="EmergencyBrake", priority=10)

        # Düşük öncelikli görev mutex'i alır
        mutex.take(t_low)
        # Yüksek öncelikli görev mutex'i ister -> Priority Inheritance devreye girer
        mutex.take(t_high)
        inherited_priority = t_low.priority  # 10 olmalıdır

        # Düşük öncelikli görev mutex'i bırakır -> Orijinal önceliğe (1) döner
        mutex.give(t_low)
        restored_priority = t_low.priority

        # 3. Deterministik 100-Tick Çizelgeleme Simülasyonu
        sched_sim = FreeRTOSScheduler()
        task_bms = sched_sim.create_task(name="BMS_EKF_1kHz", priority=8)
        task_can = sched_sim.create_task(name="CAN_RX_500Hz", priority=6)
        task_gui = sched_sim.create_task(name="UI_Dashboard", priority=2)

        for tick in range(100):
            if tick % 2 == 0:
                task_bms.state = TaskState.READY
            if tick % 4 == 0:
                task_can.state = TaskState.READY
            sched_sim.step_tick()

        kuyruk_dizi = np.array(gecikmeler_kuyruk_us)
        t_kuyruk_avg_us = float(np.mean(kuyruk_dizi))

        return {
            "kuyruk_ortalama_us": t_kuyruk_avg_us,
            "kuyruk_p99_us": float(np.percentile(kuyruk_dizi, 99)),
            "saniyelik_kuyruk_kapasitesi": int(1e6 / max(t_kuyruk_avg_us, 1e-4)),
            "inherited_priority": inherited_priority,
            "restored_priority": restored_priority,
            "context_switches_100ticks": sched_sim.context_switch_count,
            "bms_runtime_ticks": task_bms.runtime_ticks,
            "can_runtime_ticks": task_can.runtime_ticks,
            "gui_runtime_ticks": task_gui.runtime_ticks,
            "kuyruk_gecikmeler": gecikmeler_kuyruk_us[:200]
        }
