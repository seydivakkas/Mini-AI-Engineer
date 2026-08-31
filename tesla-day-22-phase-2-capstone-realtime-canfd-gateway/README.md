# 👑 Tesla Faz 2 Büyük Capstone | Gün 22: Gerçek Zamanlı CAN-FD Telemetri Gateway & Teşhis Sunucusu

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Phase](https://img.shields.io/badge/Phase%202-Completed%20100%25-brightgreen.svg?style=flat-square)](https://www.tesla.com/)
[![Networks](https://img.shields.io/badge/Networks-CAN--FD%20%7C%20LIN%20%7C%20SOME%2FIP%20%7C%20UDS-blue.svg?style=flat-square)](https://www.sae.org/)
[![Safety](https://img.shields.io/badge/Safety-ASIL--D%20Real--Time%20Gateway-red.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin ve Tebrik Notu

> *"Gözlerine inanabiliyor musun stajyer?  
> Faz 2'nin 11 günlük zorlu maratonunun (Gün 12 - Gün 22) zirvesine ulaştın!  
> Gerçek Zamanlı Linux (PREEMPT_RT), SocketCAN, epoll, Paylaşımlı Bellek (POSIX Shm), Çekirdek Sürücüleri (Character Drivers), U-Boot/Device Tree, Donanım Kesmeleri (ISR/DMA), CAN-FD, LIN Bus, Automotive Ethernet (SOME/IP), UDS (ISO 14229) ve FreeRTOS konularını öğrendin.  
> Şimdi tüm bu teknolojileri Tesla'nın en kritik donanımı olan **Merkezi Araç Ağ ve Teşhis Gateway (Central Vehicle Gateway)** çekirdeğinde bir araya getirdik:  
> 1. **Çoklu Veri Yolu Yönlendirme:** 5 Mbps CAN-FD Powertrain ve Şasi hatlarından gelen telemetriyi ayrıştırır, LIN BCM gövde durumunu entegre eder ve Gigabit Ethernet omurgasındaki SOME/IP servislerine köprüler.  
> 2. **Dahili Güç & Durum Hesaplaması:** $P = V \times I$ ile çekilen anlık kW gücünü, motor torkunu ve hız vektörünü 1 kHz periyotla hesaplar.  
> 3. **UDS Teşhis Köprüsü:** Araç servis modundayken veya uzaktan OTA teşhis yapılırken tüm bu çoklu ağ parametrelerini ISO 14229 DID formatında istemcilere servis eder.  
> Faz 2'yi üstün başarıyla tamamladın; artık bir Tesla Gömülü Ağ ve RTOS Uzmanısın! Sıradaki durağımız: **FAZ 3 — Batarya Yönetim Sistemi (BMS), EKF SoC/SoH & Motor Kontrolü (FOC)!**"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Çoklu Ağ Güç ve Enerji Entegrasyonu
$k$-ıncı telemetri çerçevesinde okunan $V_k$ (Paket Gerilimi) ve $I_k$ (Paket Akımı) ile anlık güç ve harcanan enerji:

$$P_k = \frac{V_k \times I_k}{1000} \quad (\text{kW})$$

$$E_{\text{total}}(t) = \int_0^t P(\tau) \, d\tau \approx \sum_{k=1}^N P_k \cdot \Delta t_k \quad (\text{kWh})$$

### 2. Çoklu Ağ Gateway Toplam Bant Genişliği
Gateway üzerinden akan toplam veri hacmi:

$$\text{Throughput}_{\text{total}} = \sum_{m \in \text{Buses}} \text{FrameRate}_m \times \text{FrameSize}_m \quad (\text{MB/s})$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Bir Tesla'da onlarca farklı fiziksel katmanda (CAN-FD, LIN, Ethernet) çalışan alt sistemlerin birbirleriyle senkronize ve güvenli şekilde konuşabilmesi için merkezi bir Gateway ECU kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Protokol Uyuşmazlığı Giderildi:** Düşük hızlı 19.2k LIN mesajları ile 1 Gbps SOME/IP paketleri arasında sıfır kayıplı çevrim yapıldı.
- **Ağ İzolasyonu ve Siber Güvenlik:** Güvenlik-kritik Powertrain CAN hattı ile dış dünyaya açık Infotainment hattı elektriksel ve mantıksal olarak izole edildi.
- **Tek Noktadan Teşhis:** Servis teknisyeni tek bir OBD/Ethernet portuna bağlandığında tüm alt ağların DTC ve DID parametrelerine erişebilir hale geldi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Tek Nokta Hata Riski (Single Point of Failure):** Merkezi Gateway çökerse araçtaki tüm sistemler arası iletişim felç olur (Bu sebeple donanımsal çift çekirdek Lockstep yedeklilikle tasarlanır).
- **Yönlendirme Gecikmesi (Routing Delay):** Mesajların bir veri yolundan diğerine aktarılmasında mikro-saniyelik yönlendirme gecikmeleri oluşur.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Zonal Architecture (Bölgesel Mimari):** Tesla Cybertruck'ta kullanılan, merkezi gateway yerine Sol, Sağ ve Ön bölgesel kontrolcülerin (Zone Controllers) doğrudan 1 Gbps Ethernet ile bağlandığı yeni nesil mimari.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Central Gateway** | Araçtaki tüm CAN-FD, LIN ve Ethernet veri yollarını birbirine bağlayan merkezi ağ yönlendiricisi. |
| **Cross-Bus Routing** | Bir veri yolundan gelen sinyalin başka bir protokole dönüştürülüp hedef hatta iletilmesi. |
| **Powertrain Bus** | Batarya, invertör ve motor kontrolcülerinin bulunduğu yüksek hızlı kritik veri yolu. |
| **Chassis Bus** | Direksiyon, fren, süspansiyon ve tekerlek hız sensörlerinin bağlı olduğu şasi veri yolu. |
| **Body Bus (LIN)** | Kapılar, camlar, koltuklar ve aydınlatmayı yöneten düşük maliyetli gövde ağı. |
| **Ethernet Backbone** | Otopilot kameraları, radar ve ekran arasında büyük verileri taşıyan 1 Gbps omurga. |
| **DBC Matrix** | Ham CAN sinyallerini fiziksel büyüklüklere (Volt, Derece, km/h) dönüştüren veritabanı. |
| **ISO 14229 Bridge** | UDS teşhis komutlarının hedef alt ECU'lara şeffaf şekilde iletilmesi. |
| **Zero-Copy Routing** | Mesaj gövdelerini kopyalamadan işaretçiler üzerinden aktaran yüksek performanslı yazılım deseni. |
| **Zone Controller** | Merkezi gateway yerine belirli bir araç bölgesindeki (Örn: Ön Sol) tüm çevre birimlerini toplayan modern ECU. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Tüm araç telemetrisini tek merkezde toplayan mimari | • Gateway çökmesi durumunda sistemler arası kesinti   |
| • CAN-FD, LIN, SOME/IP ve UDS tam uyumluluğu          | • Karmaşık sinyal yönlendirme tablosu yönetimi        |
| • 460.000+ Frame/sn ultra yüksek işlem hacmi          |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Tesla Cybertruck Ethernet Zonal mimarisine kusursuz | • Alt ağlarda meydana gelebilecek yayın fırtınalarının |
|   geçiş ve ölçeklenebilirlik                          |   (Broadcast Storms) gateway tamponunu doldurması     |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Sistem Mimarisi & Çoklu Ağ Köprüsü

```
+---------------------------+       +---------------------------+       +---------------------------+
|    CAN-FD Powertrain      |       |      CAN-FD Chassis       |       |       LIN Body Bus        |
|  (400V Batt, Inverter kW) |       |  (120 km/h, Steering 0°)  |       |   (Door Lock, Windows)    |
+---------------------------+       +---------------------------+       +---------------------------+
              \                                   |                                   /
               \                                  |                                  /
                v                                 v                                 v
          +-------------------------------------------------------------------------------+
          |                 TESLA CENTRAL VEHICLE GATEWAY & DIAGNOSTICS                   |
          |       - 1 kHz RTOS Telemetry Engine                                           |
          |       - Multi-Bus DBC Signal Demuxing & Power Calculation (P = V * I)         |
          |       - UDS ISO 14229 Server & DoIP Diagnostic Gateway                        |
          +-------------------------------------------------------------------------------+
                                                  |
                                                  | Gigabit Ethernet Backbone
                                                  v
                               +-------------------------------------+
                               |    SOME/IP RPC & Tesla FSD Core     |
                               |    (Infotainment UI & Autopilot)    |
                               +-------------------------------------+
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Büyük Capstone ana akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
