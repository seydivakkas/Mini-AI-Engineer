# 🚗 Tesla Gömülü Çekirdek | Gün 20: FreeRTOS Çekirdek Yapısı, Görev Senkronizasyonu & Kuyruklar

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Kernel](https://img.shields.io/badge/RTOS-FreeRTOS%20v10.5-orange.svg?style=flat-square)](https://www.freertos.org/)
[![Scheduler](https://img.shields.io/badge/Scheduler-Preemptive%20%2F%20Priority--Based-green.svg?style=flat-square)](https://www.freertos.org/)
[![Safety](https://img.shields.io/badge/Safety-Priority%20Inheritance%20Mutex-red.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"20. günümüze hoş geldin stajyer!  
> Tesla'nın batarya yönetim modüllerinde (BMB), kapı kontrol ünitelerinde ve fren mikrodenetleyicilerinde (MCU) tam teşekküllü Linux çalıştıramazsınız. Birkaç yüz kilobayt RAM'e sahip ARM Cortex-M veya TriCore mikrodenetleyicilerinde mikrosaniye hassasiyetinde determinizm gerekir.  
> İşte burada sahneye **FreeRTOS** girer:  
> 1. **Preemptive Görev Çizelgelemesi:** En yüksek öncelikli görev (Örn: Acil Frenleme veya Batarya Aşırı Akım Koruması) hazır olduğu anda işlemciyi anında devralır.  
> 2. **SysTick Zamanlayıcısı (1 kHz):** Her 1 milisaniyede bir donanım kesmesi tetiklenerek görevlerin zaman dilimleri (time slicing) ve bekleme süreleri yönetilir.  
> 3. **Thread-Safe Kuyruklar (Queues):** Görevler arasında veri aktarırken kilitlenme ve yarış durumlarını (Race Conditions) sıfıra indirir.  
> 4. **Öncelik Mirası (Priority Inheritance):** 1997 yılındaki meşhur Mars Pathfinder felaketinde olduğu gibi, düşük öncelikli bir görevin paylaşılan bir kaynağı (Mutex) tutarken yüksek öncelikli görevi engellemesini önler. Düşük öncelikli görev, yüksek öncelikli görevin yetkisini geçici olarak devralıp işini bitirir ve kaynağı serbest bırakır.  
> Bugün FreeRTOS çekirdeğini ve senkronizasyon araçlarını sıfırdan inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Gerçek Zamanlı Tepki Süresi Analizi (Response Time Analysis - RTA)
$i$ numaralı bir gerçek zamanlı görevin en kötü durum tamamlanma süresi $R_i$:

$$R_i = C_i + B_i + \sum_{j \in hp(i)} \left\lceil \frac{R_i}{T_j} \right\rceil C_j$$

Burada:
- $C_i$: Görevin salt icra süresi (Worst-Case Execution Time - WCET)
- $B_i$: Düşük öncelikli görevler tarafından Mutex ile engellenme süresi (Blocking Time)
- $hp(i)$: $i$ görevinden daha yüksek önceliğe sahip görevler kümesi
- $T_j$: Yüksek öncelikli periyodik görevlerin periyodu

### 2. Öncelik Mirası Protokolü (Priority Inheritance Protocol)
Düşük öncelikli görev $T_L$, bir Mutex $M$'i elinde tutarken yüksek öncelikli görev $T_H$ aynı Mutex'i talep ederse:

$$\text{Prio}_{\text{dyn}}(T_L) = \max\Big(\text{Prio}_{\text{base}}(T_L), \max_{k \in \text{Waiters}(M)} \text{Prio}(T_k)\Big)$$

$T_L$ Mutex'i bıraktığı (`give`) anda:

$$\text{Prio}_{\text{dyn}}(T_L) \leftarrow \text{Prio}_{\text{base}}(T_L)$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Düşük maliyetli ve kısıtlı kaynaklara sahip otomotiv mikrodenetleyicilerinde (MCU) deterministik, mikrosaniye seviyesinde zamanlanmış görev icrası sağlamak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Deterministik Tepki Süresi:** Süper döngülü (`while(1)`) spagetti kodlar yerine öncelik tabanlı çoklu görev (multitasking) mimarisi sağlandı.
- **Priority Inversion Çözümü:** Öncelik Mirası Mutex ile kritik görevlerin orta öncelikli görevler tarafından sonsuza kadar bekletilmesi engellendi.
- **Güvenli IPC:** Sıfır kopyalı ve kilitlenme korumalı FreeRTOS Queue mekanizmasıyla görevler arası telemetri aktarımı güvenceye alındı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Bellek Koruması (MPU) Zorunluluğu:** Standart FreeRTOS'ta tüm görevler aynı bellek uzayını paylaşır; bir görevin yığın taşması (Stack Overflow) tüm sistemi çökertebilir (FreeRTOS-MPU gerektirir).
- **Çok Çekirdek (SMP) Kısıtlamaları:** Klasik FreeRTOS tek çekirdek odaklıdır; çok çekirdekli sistemlerde çekirdekler arası kilitleme ek yükü oluşur.

### 4. Alternatifler Nelerdir? (Alternatives)
- **AUTOSAR Classic OS / OSEK-VDX:** Otomotivde en yaygın statik yapılandırılmış RTOS'tur; ancak konfigürasyonu FreeRTOS'a göre son derece karmaşıktır.
- **Zephyr RTOS:** Linux Vakfı tarafından geliştirilen modern, açık kaynaklı ve zengin sürücü desteğine sahip yeni nesil RTOS.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **FreeRTOS** | Gömülü mikrodenetleyiciler için açık kaynaklı, gerçek zamanlı işletim sistemi çekirdeği. |
| **Preemptive Scheduling** | Yüksek öncelikli bir görev hazır olduğunda mevcut çalışan görevin zorla durdurulup işlemcinin devredilmesi. |
| **TCB (Task Control Block)** | Görevin yığın işaretçisini, önceliğini, durumunu ve kimliğini saklayan çekirdek veri yapısı. |
| **SysTick** | İşlemci çekirdeğinde her milisaniyede bir kesme üreten donanımsal sistem zamanlayıcısı. |
| **Priority Inversion** | Düşük öncelikli görevin tuttuğu kaynak yüzünden yüksek öncelikli görevin orta öncelikli görevler tarafından dolaylı olarak engellenmesi. |
| **Priority Inheritance** | Mutex tutan düşük öncelikli görevin önceliğinin, kaynağı bekleyen en yüksek görevin seviyesine geçici olarak yükseltilmesi. |
| **Context Switch** | Bir görevin register ve yığın durumunun kaydedilip bir sonraki görevin yüklenmesi süreci. |
| **Thread-Safe Queue** | Görevler veya kesmeler arasında kesintisiz ve yarış durumsuz veri taşıyan FIFO kuyruğu. |
| **Binary Semaphore** | Görevler arasında senkronizasyon sağlayan, 0 veya 1 değerini alan bayrak mekanizması. |
| **vTaskDelay** | Görevi belirli sayıda sistem tick'i boyunca bloke (BLOCKED) duruma geçiren FreeRTOS API çağrısı. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Yalnızca birkaç kilobaytlık ultra hafif ROM/RAM izi  | • Gelişmiş sanal bellek (MMU) desteğinin bulunmaması  |
| • Deterministik 1 kHz SysTick zamanlama hassasiyeti   | • Çok çekirdekli (SMP) işlemcilerde karmaşık kilitleme|
| • Dahili Priority Inheritance ile kilitlenme önleme   |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Tesla BMB (Battery Monitoring Boards) ve kapı alt   | • Yığın taşması (Stack Overflow) durumunda hafıza     |
|   sistemlerinde sıfır maliyetli RTOS standardizasyonu |   bozulması riski                                     |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Sistem Mimarisi & Görev Durum Geçişleri

```
                +-------------------+
                |                   | (vTaskSuspend)
                |     SUSPENDED     | <---------------+
                |                   |                 |
                +-------------------+                 |
                          | (vTaskResume)             |
                          v                           |
       (xTaskCreate)   +-------------------+          |
    -----------------> |                   |          |
                       |       READY       | ---------+
                       |                   |
                       +-------------------+
                          |            ^
           (Scheduler     |            | (Yield / Preempt)
            dispatches)   v            |
                       +-------------------+
                       |                   |
                       |      RUNNING      |
                       |                   |
                       +-------------------+
                          |            ^
          (vTaskDelay /   |            | (Timeout expired /
           Queue empty)   v            |  Mutex given)
                       +-------------------+
                       |                   |
                       |      BLOCKED      |
                       |                   |
                       +-------------------+
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana çizelgeleme akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
