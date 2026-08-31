# 🚗 Tesla FSD Otonom Sürüş | Gün 70: Araç İçi Ses Boru Hattı: PipeWire/ALSA, Aktif Gürültü Engelleme (ARNC) ve Çok Bölgeli Ses

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Audio](https://img.shields.io/badge/Audio-PipeWire%2048%20kHz%20Low--Latency-red.svg?style=flat-square)](https://www.tesla.com/)
[![ARNC](https://img.shields.io/badge/DSP-Active%20Road%20Noise%20Cancellation-blue.svg?style=flat-square)](https://pipewire.org/)
[![Performance](https://img.shields.io/badge/Acoustic-%3E15%20dB%20Noise%20Reduction-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"70. günümüze hoş geldin stajyer!  
> Elektrikli bir araçta içten yanmalı motor sesi olmadığı için yoldan ve lastiklerden gelen asfalt uğultusu (Road Noise) çok daha belirgin hale gelir.  
> Tesla Model S/X ve Model 3 Highland'de bu sorunu çözmek için **Aktif Yol Gürültüsü Engelleme (ARNC - Active Road Noise Cancellation)** teknolojisi kullanılır:  
> 1. **Tekerlek Yuvası İvmeölçerleri:** Lastiğin yol yüzeyine çarpmasıyla oluşan titreşimler anında elektrik sinyaline ($x(t)$) dönüştürülür.  
> 2. **180 Derece Ters Faz Üretimi (Anti-Noise):** DSP çekirdeği gürültünün tam zıt fazındaki ses dalgasını ($y(t) = -x(t)$) üretir.  
> 3. **Akustik Yıkıcı Girişim (Destructive Interference):** Kapı ve koltuk hoparlörlerinden yayılan anti-gürültü ile yol gürültüsü havada çarpışarak birbirini yok eder ($>15\text{ dB}$ sessizlik).  
> 4. **PipeWire Çok Bölgeli Ses:** Otopilot ve navigasyon sesleri sadece sürücünün koltuk başlığına giderken, arka koltuktaki yolcular kulaklıklarından film izleyebilir.  
> Bugün Tesla kabinini kütüphane sessizliğine kavuşturan ARNC ses motorunu kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Akustik Yıkıcı Girişim ve Ters Faz (Destructive Interference)

$$x(t) = A \sin(\omega t + \phi), \quad y(t) = -x(t) = A \sin(\omega t + \phi + \pi)$$

$$r(t) = x(t) + y(t) \to 0$$

### 2. Desibel Cinsinden Ses Basıncı Sönümleme Oranı

$$\Delta \text{dB} = 10 \log_{10}\left( \frac{P_{\text{raw}}}{P_{\text{residual}}} \right) = 10 \log_{10}\left( \frac{\frac{1}{N} \sum_{i=1}^N x[i]^2}{\frac{1}{N} \sum_{i=1}^N r[i]^2} \right) \ge 12.0\text{ dB}$$

### 3. PipeWire Tampon Gecikmesi (Buffer Latency)

$$t_{\text{latency}} = \frac{N_{\text{buffer}}}{f_s} \times 1000\text{ ms} = \frac{64}{48,000\text{ Hz}} \times 1000 \approx 1.333\text{ ms}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Fiziksel yalıtım malzemeleri aracı ağırlaştırıp menzili düşürdüğü için, yazılımsal ve dijital DSP sinyal işleme ile aktif kabin sessizliği sağlamak amacıyla kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Düşük Frekanslı Yol Uğultusu ($50-300\text{ Hz}$):** Ağır süngerlerin ve camların engelleyemediği düşük frekanslı yol gürültüsünü hoparlörlerden yok etti.
- **Ses Çakışması:** Sürücünün navigasyon uyarıları ile kabindeki müzik sesini bağımsız yönlendirdi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Faz Gecikmesi:** Ses dalgası hoparlörden kulağa ulaşana kadar geçen süre milimetrik kalibre edilmezse ters faz yerine yapıcı girişim (gürültü artışı) olabilir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Pasif Akustik Yalıtım (Sünger/Kurşun Plaka):** Araca 40+ kg ağırlık ekler, menzili %5 düşürür.
- **PulseAudio (Eski):** Yüksek tampon gecikmesi ($>20\text{ ms}$) nedeniyle gerçek zamanlı ters faz üretemez.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **ARNC (Active Road Noise Cancellation)**| Yol ve lastik gürültüsünü ters ses dalgaları yayarak kabin içinde yok eden sistem. |
| **Anti-Noise** | Gürültü sinyaliyle aynı genlikte fakat $180^\circ$ ($\pi$ radyan) ters fazda üretilen karşıt ses dalgası. |
| **Destructive Interference**| İki zıt fazlı ses dalgasının birbirinin tepe ve çukur noktalarını söndürmesi olayı. |
| **PipeWire** | Modern Linux çekirdeğinde ultra düşük gecikmeli ses ve video akış sunucusu. |
| **DSP (Digital Signal Processor)**| Ses sinyallerini mikrosaniyeler içinde filtreleyen özel sayısal işlemci. |
| **Multi-Zone Audio** | Aracın farklı koltuklarına birbirinden bağımsız ses akışlarının yönlendirilmesi. |
| **Headrest Speaker** | Sürücünün koltuk başlığına gömülü özel navigasyon ve güvenlik uyarısı hoparlörü. |
| **Buffer Size** | Ses kartının her çevrimde işlediği örnek sayısı (64 örnek = $1.33\text{ ms}$). |
| **SPL (Sound Pressure Level)** | Sesin desibel (dB) cinsinden akustik şiddeti. |
| **FXLMS Filter** | Akustik gecikmeyi telafi eden Filtrelenmiş-X En Küçük Ortalama Kareler adaptif filtresi. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Ağır yalıtım malzemesi eklemeden >15 dB sessizlik  | • Akustik kabin gecikmesi için milimetrik hoparlör    |
| • 1.33 ms ultra düşük PipeWire tampon gecikmesi       |   kalibrasyonu zorunluluğu                            |
| • Kişiselleştirilmiş koltuk başlığı ses yönlendirmesi|                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Robotaksi yolcularına uçak birinci sınıf konforunda | • Açık pencerelerden giren kaotik rüzgar sesinin      |
|   gürültüsüz akustik dinlenme ortamı sunma            |   ARNC algoritmasını yanıltması                       |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla ARNC ve PipeWire Mimari Akış Şeması

```
[ Tekerlek Yuvası İvmeölçerleri (Yol Titreşimi) ]
                         |
                         v
     [ 1. Ham Gürültü Örnekleme: 48 kHz, Buffer=64 ]
                         |
                         v
     [ 2. DSP 180° Ters Faz Üretimi: y(t) = -x(t) ]
                         |
                         v
     [ 3. PipeWire Çok Bölgeli Ses Yönlendirme ]
       |                                   |
       v (Sürücü Bölgesi)                  v (Ana Kabin)
[ Koltuk Başlığı Hoparlörü ]       [ 22-Hoparlör Premium Ses ]
(Nav + Otopilot Uyarıları)         (Anti-Noise + Müzik)
                         |
                         v
         [ Kabin İçi Akustik Sönümleme (>15 dB Sessizlik) ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana ARNC Ses simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
