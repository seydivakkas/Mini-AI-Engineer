# 🚗 Tesla FSD Otonom Sürüş | Gün 86: Yüksek Frekanslı Güç Telemetrisi ve MQTT/Kafka ile Bulut Senkronizasyonu

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Telemetry](https://img.shields.io/badge/Streaming-100%20Hz%20Power%20Telemetry-red.svg?style=flat-square)](https://www.tesla.com)
[![Binary-Struct](https://img.shields.io/badge/Payload-32--Byte%20Binary%20Struct-blue.svg?style=flat-square)](https://docs.python.org/3/library/struct.html)
[![Zero-Loss](https://img.shields.io/badge/Buffer-Circular%20Ring%20Zero--Loss-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"86. günümüze hoş geldin stajyer!  
> Dünya genelindeki milyonlarca Supercharger, Powerwall ve Megapack ünitesinin şebeke dalgalanmalarını anlık olarak izleyebilmek için saniyede yüzlerce telemetri ölçümüne ihtiyaç duyarız.  
> Ancak her ünite saniyede 100 kez JSON metin paketi gönderirse ($185\text{ Bayt} \times 100\text{ Hz} = 18.5\text{ KB/s}$), hücresel veri faturaları milyonlarca dolara ulaşır ve bulut sunucuları kilitlenir!  
> Tesla bu devasa telemetri akışını **32 Baytlık Kompakt Binary Struct ve Kayan Pencere (Sliding Window)** mimarisiyle çözer:  
> 1. **100 Hz Yüksek Frekans:** Voltaj, akım, aktif güç, reaktif güç, frekans ve sıcaklık $10\text{ ms}$ aralıklarla ölçülür.  
> 2. **32 Baytlık Binary Struct:** `>Qffffff` ikili formatıyla veri boyutu $\%83$ küçültülür (Saniyede sadece $3.125\text{ KB/s}$).  
> 3. **1 Saniyelik Kayan Pencere:** Ham yüksek frekanslı sinyalin ortalama, tepe ve standart sapmasını yerel olarak hesaplar.  
> 4. **Halka Arabellek (Circular Ring Buffer):** Tünel veya hücresel baz istasyonu kesintilerinde verileri hafızada saklar; ağ geri geldiğinde sıfır veri kaybıyla buluta aktarır.  
> Bugün Tesla'nın global telemetri omurgasını oluşturan 100 Hz güç akış motorunu kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Kompakt İkili Paket Boyutu ve Bant Genişliği

$$\text{Size}_{\text{packet}} = 8\ (\text{Timestamp uint64}) + 6 \times 4\ (\text{Float32}) = 32\ \text{Bayt}$$

$$\text{Bandwidth} = 100\ \text{Hz} \times 32\ \text{Bayt} = 3,200\ \text{Bayt/sn} = 3.125\ \text{KB/sn}$$

### 2. Kayan Pencere (Sliding Window) Ortalama Güç Formülü

$$\bar{P}(t) = \frac{1}{N} \sum_{k=0}^{N-1} P(t - k \cdot \Delta t), \quad N = 100\ \text{Örnek}, \quad \Delta t = 10\ \text{ms}$$

### 3. Halka Arabellek (Circular Ring Buffer) Sıfır Kayıp Garantisi

$$B_{\text{capacity}} = 1000\ \text{Paket} \implies T_{\text{outage\_retention}} = \frac{1000}{100\ \text{Hz}} = 10.0\ \text{saniye}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Şebeke geçici durumlarını (transients), gerilim çökmelerini ve harmonik dalgalanmaları $10\text{ ms}$ çözünürlükle yakalayabilmek ve hücresel ağ maliyetlerini minimumda tutmak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **JSON Bant Genişliği Şişkinliği:** Hantal JSON metin formatını terk ederek veri iletim hacmini $\%83$ azalttı.
- **Ağ Kopmalarında Veri Kaybı:** Halka arabellek ile hücresel sinyal kopukluklarında telemetri kayıplarını sıfıra indirdi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **RAM Sınırları:** Uzun süreli ağ kesintilerinde (örneğin 1 saat) halka arabelleğin flash belleğe (eMMC/SSD) yazılması gerekir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Düşük Frekanslı Örnekleme (1 Hz):** Çok az veri harcar ancak 10 ms süren ani şebeke kısa devrelerini kaçırır.
- **REST HTTP POST:** Her istekte TCP el sıkışması ve HTTP başlık yükü yaratır (100 Hz için imkansızdır).

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Power Telemetry** | Elektriksel gerilim, akım, güç ve frekans parametrelerinin uzaktan izlenmesi. |
| **100 Hz Sampling** | Saniyede 100 kez (her 10 milisaniyede bir) analog-dijital çevrim (ADC) ile veri toplama. |
| **Binary Serialization** | Sayısal verilerin metin yerine doğrudan ikili bellek baytları halinde paketlenmesi. |
| **Sliding Window** | Sabit bir zaman aralığındaki (1 saniye) en güncel veri grubunu istatistiksel olarak işleme. |
| **Circular Ring Buffer** | Başı ve sonu birleşik, en eski verinin üzerine yazarak sonsuz döngüde çalışan bellek yapısı. |
| **MQTT QoS 1** | Mesajın en az bir kez karşı tarafa ulaştığını garanti eden hafif IoT iletişim protokolü. |
| **Apache Kafka** | Saniyede milyonlarca telemetri olayını yüksek hızda işleyen dağıtık veri akış platformu. |
| **Active Power (P)** | Cihazın iş yapan faydalı elektriksel gücü (Kilowatt - kW). |
| **Reactive Power (Q)** | Manyetik ve elektriksel alan oluşturmak için şebekeden çekilip verilen reaktif güç (kVAR). |
| **Network Jitter** | Telemetri paketlerinin ağ üzerinden varış zamanlarındaki mikrosaniyelik sapma/oynama. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • 32 bayt kompakt struct ile %83 bant genişliği tasarrufu| Uzun süreli internet kesintilerinde RAM dolması      |
| • 100 Hz yüksek çözünürlüklü şebeke arıza yakalama    |   (Flash depolama yedeği gerektirir)                  |
| • 0.8 µs RTOS paketleme döngüsü                       |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Tüm Tesla araç ve enerji filo telemetrisini tek     | • Hücresel operatörlerin baz istasyonu yoğunluğunda   |
|   merkezi Kafka kümesinde gerçek zamanlı işleme       |   paket teslimatlarını geciktirmesi                   |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Güç Telemetrisi Akış Şeması

```
[ Supercharger / Megapack Sensörleri (ADC) ]
                     |
                     | 100 Hz (10 ms) Ham Ölçüm
                     v
   [ 32-Bayt Kompakt Binary Struct Paketleyici ]
                     |
                     | [Timestamp, V, I, P, Q, f, T]
                     v
     [ 1000-Elemanlı Halka Arabellek (RAM) ]
          /                         \
         /                           \
    Ağ Bağlantısı Var            Ağ Bağlantısı Koptu
         |                               |
         v                               v
[ MQTT / Kafka Akışı ]         [ Hafızada Güvenle Sakla ]
(3.125 KB/s Buluta Aktar)      (Sıfır Veri Kaybı)
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana telemetri simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
