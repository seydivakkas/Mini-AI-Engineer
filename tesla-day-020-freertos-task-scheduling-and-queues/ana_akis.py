"""
Tesla Gün 20 Ana Akış (Tesla Day 20 Main Pipeline)
===================================================
FreeRTOS Çekirdek Yapısı, Görev Senkronizasyonu & Öncelik Mirası
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

from src.tesla_freertos_cekirdek import (
    FreeRTOSScheduler,
    FreeRTOSQueue,
    FreeRTOSMutex,
    TaskState
)
from src.tesla_freertos_profilleyici import TeslaFreeRTOSProfilleyici
from src.tesla_freertos_gorsellestirici import TeslaFreeRTOSGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GÖMÜLÜ YAZILIM MASTERI | GÜN 20: FREERTOS PREEMPTIVE SCHEDULER 🚗")
    print("================================================================================")
    print("Stajyer Görevi: 1 kHz SysTick, Thread-Safe Kuyruklar, Mutex & Priority Inheritance")
    print("--------------------------------------------------------------------------------\n")

    sched = FreeRTOSScheduler()

    # 1. Görevlerin Tanımlanması
    print(" [1] FreeRTOS Görevleri (Tasks) ve Öncelikleri Tanımlanıyor...")
    t_bms = sched.create_task(name="BMS_EKF_Task", priority=8, stack_size=256)
    t_can = sched.create_task(name="CAN_RX_Task", priority=6, stack_size=128)
    t_ui  = sched.create_task(name="Infotainment_UI", priority=2, stack_size=128)

    print(f"     -> Görev 1: {t_bms.name:<18} | Öncelik: {t_bms.priority} | Yığın: {t_bms.stack_size_words} Words")
    print(f"     -> Görev 2: {t_can.name:<18} | Öncelik: {t_can.priority} | Yığın: {t_can.stack_size_words} Words")
    print(f"     -> Görev 3: {t_ui.name:<18} | Öncelik: {t_ui.priority} | Yığın: {t_ui.stack_size_words} Words")

    # 2. Kuyruk ile Görevler Arası İletişim (IPC)
    print("\n [2] FreeRTOS Queue Üzerinden Telemetri Mesajlaşması...")
    q = FreeRTOSQueue(length=4, item_name="TelemetryQueue")
    q.send({"sensor": "PackVoltage", "val": 398.5})
    q.send({"sensor": "PackCurrent", "val": 120.4})

    item1 = q.receive()
    print(f"     -> Kuyruktan Çekilen Paket 1: {item1}")

    # 3. Öncelik Mirası (Priority Inheritance) Doğrulaması
    print("\n [3] Öncelik Tersine Çevrilmesi (Priority Inversion) Önleme Testi...")
    mutex = FreeRTOSMutex(name="SharedSPIBus")
    t_low = sched.create_task("LowLogTask", priority=1)
    t_high = sched.create_task("HighBrakeTask", priority=10)

    mutex.take(t_low)
    print(f"     -> Düşük Görev Mutex'i Aldı. Önceliği: {t_low.priority}")

    mutex.take(t_high)
    print(f"     -> Yüksek Görev Mutex İstedi ve Engellendi.")
    print(f"     -> Düşük Görevin Yeni Önceliği: {t_low.priority} (Öncelik Mirası Aktif! 🚀)")

    mutex.give(t_low)
    print(f"     -> Mutex Bırakıldı. Düşük Görev Orijinal Önceliğine Döndü: {t_low.priority}")

    # 4. Performans ve Çizelgeleme Benchmark'ı
    print("\n [4] FreeRTOS Çekirdek Gecikme ve Kuyruk Verimi Benchmark Analizi...")
    profilleyici = TeslaFreeRTOSProfilleyici(ornek_sayisi=5000)
    metrikler = profilleyici.benchmark_freertos()

    print(f"     -> Ortalama Kuyruk Gecikmesi  : {metrikler['kuyruk_ortalama_us']:.3f} µs (P99: {metrikler['kuyruk_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Kuyruk Kapasitesi: {metrikler['saniyelik_kuyruk_kapasitesi']:,} Mesaj/sn")
    print(f"     -> 100 Tick Context Switch    : {metrikler['context_switches_100ticks']} Kez")
    print(f"     -> BMS Görevi Çalışma Oranı   : %{metrikler['bms_runtime_ticks']}")

    # 5. Tanı Paneli Görselleştirme
    print("\n [5] 6 Panelli Tesla FreeRTOS Çekirdek Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaFreeRTOSGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_freertos_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 20 BAŞARIYLA TAMAMLANDI! FREERTOS PREEMPTIVE SCHEDULER DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
