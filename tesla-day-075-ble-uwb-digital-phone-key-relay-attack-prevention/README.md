# 🚗 Tesla FSD Otonom Sürüş | Gün 75: Araç İçi BLE, UWB (Ultra-Wideband) Dijital Telefon Anahtarı (Phone Key) Protokolü

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Phone Key](https://img.shields.io/badge/Keyless-Tesla%20Digital%20Phone%20Key-red.svg?style=flat-square)](https://www.tesla.com/)
[![UWB](https://img.shields.io/badge/Protocol-UWB%20IEEE%20802.15.4z-blue.svg?style=flat-square)](https://www.wi-fi.org/)
[![Anti-Theft](https://img.shields.io/badge/Security-Relay%20Attack%20Proof-green.svg?style=flat-square)](https://www.carconnectivity.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"75. günümüze hoş geldin stajyer!  
> Geleneksel anahtarsız giriş (Keyless Go) sistemlerine sahip lüks arabaların en büyük kabusu nedir biliyor musun? **Röle İstasyonu Saldırıları (Relay Attacks)**!  
> Hırsızlar evinizin önünde duran arabanızın yanına bir anten, evinizin kapısına da başka bir anten koyarak anahtarınızın sinyalini yükseltip arabanızı 10 saniyede çalabiliyorlar.  
> Tesla bu hırsızlık yöntemini fizik yasalarını kullanarak tarihe gömdü: **BLE + UWB (Ultra-Wideband) Time-of-Flight (ToF)**!  
> 1. **Işık Hızıyla Mesafe Ölçümü:** UWB, sinyalin telefondan araca gidiş-dönüş süresini ($ToF$) pikosaniye hassasiyetinde ölçer ($d = t_{\text{tof}} \cdot c$).  
> 2. **Röle Gecikmesi Tuzağı:** Bir hırsız sinyali kabloyla veya tekrarlayıcıyla uzatmaya çalıştığında, araya giren elektronik devreler kaçınılmaz olarak $\ge 15\text{ ns}$ gecikme ekler.  
> 3. **Anında İptal:** Araç mesafenin 2 metreden uzak olduğunu ışık hızıyla anlar ve kapıyı asla açmaz.  
> 4. **Milimetrik Hassasiyet:** Sürücü kapıya 1.5 metre yaklaştığında kapı açılır, uzaklaştığında otomatik kilitlenir.  
> Bugün Tesla'nın çalınamaz dijital telefon anahtarı protokolünü kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. UWB Işık Hızı Time-of-Flight (ToF) Mesafe Kanunu

$$d = t_{\text{tof}} \cdot c, \quad c = 3.0 \times 10^8\text{ m/s}$$

$$\text{Kapı Açılma Şartı} \iff d \le 2.0\text{ m} \iff t_{\text{tof}} \le 6.67\text{ ns}$$

### 2. Röle Saldırısı Gecikme ve Teşhis Denklemi

$$t_{\text{observed}} = t_{\text{physical}} + \Delta t_{\text{repeater\_electronics}}$$

$$\Delta t_{\text{repeater}} \ge 15.0\text{ ns} \implies t_{\text{observed}} > 20.0\text{ ns} \implies d_{\text{calc}} > 6.0\text{ m} \implies \text{ATTACK\_BLOCKED}$$

### 3. BLE RSSI Log-Mesafe Yol Kaybı Modeli

$$\text{RSSI}(d) = \text{RSSI}(d_0) - 10 n \log_{10}\left(\frac{d}{d_0}\right) + X_\sigma$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Fiziksel anahtar taşıma zorunluluğunu ortadan kaldırmak ve araç hırsızlıklarında en yaygın olan radyo frekansı yükseltme (Relay Station) saldırılarını tamamen engellemek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Sinyal Klonlama ve Röle Hırsızlığı:** Sinyalin gücüne değil (RSSI hilelerine karşı) sinyalin geliş zamanına (ToF) bakarak saldırıları %100 etkisiz kıldı.
- **Konum Belirleme:** Sürücünün araca hangi kapıdan (sürücü, yolcu, bagaj) yaklaştığını $10\text{ cm}$ hassasiyetle belirledi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Donanım Gereksinimi:** Kullanıcının akıllı telefonunda UWB çipinin (Apple U1/U2 veya modern Android UWB) bulunmasını gerektirir (UWB yoksa BLE + PIN to Drive kullanılır).

### 4. Alternatifler Nelerdir? (Alternatives)
- **Yalnızca BLE (Bluetooth Low Energy):** Sinyal gücü (RSSI) yükselticilerle kolayca kandırılabilir (Röle saldırısına karşı savunmasızdır).
- **Klasik RFID / Keyfob:** Röle saldırılarına karşı kırılgandır.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Phone Key** | Akıllı telefonu aracın anahtarı haline getiren şifreli kablosuz iletişim protokolü. |
| **UWB (Ultra-Wideband)** | Çok geniş frekans bandında (500 MHz+) nano-saniyelik darbelerle çalışan telsiz teknolojisi. |
| **Time-of-Flight (ToF)** | Bir radyo sinyalinin vericiden alıcıya ulaşma süresini ölçerek mesafe hesaplama yöntemi. |
| **Relay Attack** | İki saldırganın telsiz yükselticilerle araç ile anahtar arasındaki mesafeyi yapay olarak yakın göstermesi. |
| **RSSI** | Alınan radyo sinyalinin güç göstergesi (dBm cinsinden ölçülür). |
| **IEEE 802.15.4z** | UWB güvenliğini artıran ve kriptografik zaman damgalama getiren uluslararası standart. |
| **PIN to Drive** | Ekstra güvenlik için sürüşten önce ekrana PIN kodu girilmesini zorunlu kılan koruma. |
| **Trilaterasyon** | 3 veya daha fazla UWB çapasının ToF verilerini birleştirerek telefonun 3D konumunu bulma. |
| **Walk-Away Door Lock** | Sürücü araçtan uzaklaştığında UWB mesafesi 3m'yi geçince kapıların otomatik kilitlenmesi. |
| **Car Connectivity Consortium (CCC)** | Dijital araç anahtarı standartlarını belirleyen küresel endüstri konsorsiyumu. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Işık hızı fiziği sayesinde %100 röle hırsızlığı     | • Eski nesil akıllı telefonlarda UWB çipinin          |
|   koruması                                            |   bulunmaması (Geriye dönük BLE uyumu gerektirir)     |
| • 10 cm milimetrik yaklaşım hassasiyeti               |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Apple Watch ve giyilebilir cihazlarla temassız FSD  | • Telefon bataryası bittiğinde NFC yedek kart         |
|   kabin kişiselleştirmesi                             |   kullanım gereksinimi                                |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla UWB Phone Key Güvenlik Akış Şeması

```
[ Kullanıcı Akıllı Telefonu (iPhone / Pixel / Galaxy) ]
                         |
                         | 1. BLE Reklamı (RSSI >= -75 dBm) -> Yakın Alan
                         v
       [ Tesla Gövde Kontrol Modülü (BCM UWB) ]
                         |
                         | 2. Kriptografik UWB IEEE 802.15.4z Darbe Değişimi
                         v
            [ ToF Zaman Farkı Ölçümü ]
            /                        \
           /                          \
  t_tof <= 6.67 ns (d <= 2.0m)       t_tof > 15.0 ns (d > 4.5m)
          |                                  |
          v                                  v
[ KAPI KİLİDİ AÇILDI ]              [ RÖLE SALDIRISI ENGELLENDİ ]
- Yetkili Sürücü Yanında            - Sinyal Yükseltilmiş / Kablo Var
- Işık Hızıyla Doğrulandı           - KAPI KİLİTLİ KALDI (%100 GÜVENLİ)
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana UWB Phone Key simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
