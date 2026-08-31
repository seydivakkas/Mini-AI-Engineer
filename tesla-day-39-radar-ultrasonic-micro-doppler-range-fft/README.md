# 🚗 Tesla FSD Otonom Sürüş | Gün 39: Ultrasonik ve Milimetrik Radar Sinyal İşleme (Micro-Doppler ve Range-FFT)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Radar](https://img.shields.io/badge/Radar-77%20GHz%20FMCW%20Chirp-red.svg?style=flat-square)](https://www.tesla.com/)
[![FFT](https://img.shields.io/badge/Signal-2D%20Range--Doppler%20FFT-blue.svg?style=flat-square)](https://www.sae.org/)
[![Ultrasonic](https://img.shields.io/badge/Sonar-ToF%20Temp%20Compensated-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"39. günümüze hoş geldin stajyer!  
> Tesla bugün ağırlıklı olarak Tesla Vision (kameralar) kullansa da, önceki nesil HW2.5/HW3 araçlardaki ve Cybertruck'taki yüksek çözünürlüklü **Phoenix Radar** ile tampon park sensörlerinin sinyal işleme prensipleri her otomotiv yazılım mühendisinin uzmanlaşması gereken temel konulardır:  
> 1. **77 GHz FMCW (Frekans Modülasyonlu Sürekli Dalga) Radar:** Frekansı zamanla doğrusal artan (Chirp) bir mikrodalga sinyali yayar. Yansıyan sinyalle orijinal sinyal karıştırıldığında (Mixer) aradaki fark frekansı (Beat Frequency $f_b$) mesafeyi, faz kayması ise Doppler hızını ($f_d$) verir.  
> 2. **2D Range-Doppler FFT:** Hızlı zamanda (Fast-Time) 1D FFT mesafeyi ($R$), yavaş zamanda (Slow-Time) 2D FFT bağıl hızı ($v$) çözerek $64 \times 256$ boyutlu 2D Güç Haritası üretir.  
> 3. **CA-CFAR (Constant False Alarm Rate):** Gürültü seviyesini komşu eğitim hücrelerinden dinamik hesaplayarak sahte hedefleri (Phantom Targets) engeller.  
> 4. **Micro-Doppler İmzası:** Yürüyen bir yayanın kolları ve bacakları gövdeden farklı hızda salındığı için Doppler spektrumunda periyodik modülasyon desenleri oluşturur (Yaya sınıflandırıcısı).  
> 5. **Ultrasonik Park Sensörü (ToF):** Ses dalgasının gidiş-dönüş süresinden ($t_{\text{echo}}$) mesafe hesaplanır; hava sıcaklığına göre ses hızı düzeltmesi yapılır.  
> Bugün radar ve sonar sinyal işleme çekirdeğini kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. FMCW Beat Frekansı ve Mesafe Denklemi

$$f_b = \frac{2 \cdot S \cdot R}{c} \implies R = \frac{c \cdot f_b \cdot T_c}{2 \cdot B}$$

### 2. Doppler Frekansı ve Bağıl Hız

$$f_d = \frac{2 \cdot v_{\text{rel}}}{\lambda} \implies v_{\text{rel}} = \frac{c \cdot f_d}{2 \cdot f_c}$$

### 3. CA-CFAR Dinamik Eşikleme

$$T_{\text{CFAR}} = \frac{1}{N_{\text{train}}} \sum_{k \in \text{Train}} P_k + \alpha_{\text{offset}}$$

### 4. Sıcaklık Kompanzasyonlu Ultrasonik Mesafe

$$v_{\text{sound}}(T) = 331.3 \cdot \sqrt{1 + \frac{T(^\circ\text{C})}{273.15}} \quad \text{m/s}$$

$$d_{\text{ultrasonic}} = \frac{v_{\text{sound}}(T) \cdot t_{\text{echo}}}{2}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Görüşün sıfıra indiği yoğun sis, kar fırtınası veya toz bulutlarında dahi elektromanyetik dalgalarla öndeki araçların mesafesini ve hızını ölçmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Sis ve Yağmur Körlüğü:** Kameraların göremediği durumlarda 77 GHz milimetrik dalgaların su damlacıklarından etkilenmeden hedefleri yakalamasını sağladı.
- **Doğrudan Hız Ölçümü:** Optik türev almaya gerek kalmadan Doppler kaymasından bağıl hızı anında ölçtü.
- **Dinamik Gürültü Eşikleme:** CA-CFAR ile zemin yansımalarını süzerek sahte frenlemeleri önledi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Açısal Çözünürlük (Angular Resolution):** Radar, nesnelerin yatay sınırlarını ve şerit çizgilerini kameralar kadar keskin ayıramaz.
- **Metal Köprüler ve Levhalar:** Durgun metal nesneler devasa radar yankısı üreterek yanlış fren tetikleyebilir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Saf Görsel Optik Akış (Tesla Vision):** Donanım maliyetini düşürür ancak aşırı siste zayıflar.
- **LiDAR:** Çok yüksek açısal çözünürlük sağlar ancak pahalıdır ve siste performansı düşer.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **FMCW Radar** | Frekansı zamanla doğrusal değişen sürekli dalgalarla mesafe ve hız ölçen radar türü. |
| **Chirp Sinyali** | Belirli bir süre ($T_c$) boyunca frekansı $f_0$'dan $f_0 + B$'ye artan mikrodalga darbesi. |
| **Beat Frequency ($f_b$)** | İletilen ve yansıyan radar sinyallerinin mikser çıkışındaki fark frekansı ($Hz$). |
| **Range-FFT** | Hızlı zaman örneklerine uygulanan ve mesafe spektrumunu çıkaran ilk 1D Fourier dönüşümü. |
| **Doppler-FFT** | Ardışık chirp darbeleri boyunca uygulanan ve bağıl hız spektrumunu çıkaran ikinci 1D FFT. |
| **CA-CFAR** | Hücre ortalamalı gürültü tahminiyle sabit yanlış alarm oranı sağlayan adaptif eşikleyici. |
| **Micro-Doppler** | Hedefin ana gövdesi dışındaki hareketli parçalarının (kollar, bacaklar, tekerlekler) oluşturduğu frekans modülasyonu. |
| **Time-of-Flight (ToF)** | Ultrasonik veya lazer darbesinin hedefe ulaşıp geri dönme süresi ($s$). |
| **Guard Cells** | CFAR eşiklemesinde hedef pikinin gürültü hesabına karışmasını önleyen güvenlik hücreleri. |
| **Phoenix Radar** | Tesla'nın yüksek çözünürlüklü 4D görüntüleme radarı geliştirme projesi. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Sis, toz ve karanlıkta %100 güvenilir mesafe/hız    | • Düşük açısal çözünürlük ve şerit görememe           |
| • 220 µs hızlı 2D FFT çözümleme kapasitesi            | • Metal köprülerde sahte yüksek yankı üretme          |
| • Sıcaklık kompanzasyonlu ultrasonik park desteği     |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • 4D Görüntüleme Radarları (Imaging Radar) ile        | • Çapraz araçlardan gelen aktif radar sinyali         |
|   nokta bulutu seviyesinde yüksek çözünürlük          |   karışması (Interference)                            |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ 77 GHz FMCW Radar ve Ultrasonik Sinyal Akışı

```
[ 77 GHz FMCW TX Sinyali ] ===(Hedefe Çarpma)===> [ RX Karıştırıcı (Mixer) ]
                                                          |
                                                          v
                                               [ 1D Range-FFT (Mesafe) ]
                                                          |
                                                          v
                                               [ 2D Doppler-FFT (Hız) ]
                                                          |
                                                          v
                                               [ CA-CFAR Dinamik Eşikleme ]
                                                          |
                                                          v
                                             [ 3D Hedef: Range & Velocity ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana radar ve ultrasonik sinyal işleme akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
