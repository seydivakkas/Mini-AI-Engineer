# 🚗 Tesla FSD Otonom Sürüş | Gün 62: Acil Durum Manevraları ve Otomatik Acil Frenleme (AEB) Kontrol Mantığı

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![AEB](https://img.shields.io/badge/Safety-Euro--NCAP%20AEB%20Protocol-red.svg?style=flat-square)](https://www.tesla.com/)
[![Stopping Dist](https://img.shields.io/badge/Physics-Emergency%20Stopping%20Distance-blue.svg?style=flat-square)](https://www.sae.org/)
[![Evasive Steer](https://img.shields.io/badge/Control-Autonomous%20Emergency%20Steering-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"62. günümüze hoş geldin stajyer!  
> Otonom araç yazılımının en kritik koruma kalkanı **Otomatik Acil Frenleme (AEB - Autonomous Emergency Braking)** ve **Acil Kaçınma Direksiyonu (AES - Autonomous Emergency Steering)** sistemleridir.  
> Bu sistem her milisaniyede aracın önündeki engelleri ve durma mesafesini fizik yasalarıyla hesaplar:  
> 1. **Acil Durma Mesafesi ($d_{\text{stop}} = v \cdot t_{\text{delay}} + \frac{v^2}{2 a_{\max}}$):** Aktüatör gecikmesi ($0.2\text{ s}$) ve maksimum fren ivmesi ($9.0\text{ m/s}^2 \approx 0.92g$) hesaba katılır.  
> 2. **Euro-NCAP Kademeli Müdahale (Tier System):**  
>    - **FCW ($TTC \le 2.4\text{s}$):** Sürücüye sesli ve görsel çarpışma uyarısı.  
>    - **Kısmi Fren ($TTC \le 1.6\text{s}$):** $-4.0\text{ m/s}^2$ ön frenleme ile balata boşluğu alınır.  
>    - **Tam AEB ($TTC \le 1.0\text{s}$):** $-9.0\text{ m/s}^2$ maksimum hidrolik frenleme.  
> 3. **Acil Kaçınma Direksiyonu (AES):** Eğer engel mesafesi durma mesafesinden kısaysa ve yan şerit boşsa, araç yalnızca fren yapmak yerine direksiyonu kırarak engelin etrafından sıyrılır.  
> Bugün Tesla'nın Euro-NCAP 5-Yıldız güvenlik puanının arkasındaki hayat kurtaran AEB algoritmasını kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Acil Durum Durma Mesafesi Formülasyonu

$$d_{\text{stop}}(v) = v \cdot t_{\text{delay}} + \frac{v^2}{2 a_{\max}}, \quad t_{\text{delay}} = 0.20\text{ s}, \ a_{\max} = 9.0\text{ m/s}^2$$

### 2. Time-To-Collision (TTC)

$$TTC = \frac{d_{\text{obstacle}}}{v_{\text{rel}}}$$

### 3. Euro-NCAP Müdahale Kademeleri

$$\text{Müdahale} = \begin{cases} \text{NORMAL}, & TTC > 2.4\text{ s} \\ \text{FCW UYARISI}, & 1.6\text{ s} < TTC \le 2.4\text{ s} \\ \text{KISMİ FREN } (-4.0\text{ m/s}^2), & 1.0\text{ s} < TTC \le 1.6\text{ s} \\ \text{TAM AEB } (-9.0\text{ m/s}^2), & TTC \le 1.0\text{ s} \text{ veya } d_{\text{obs}} \le d_{\text{stop}} \end{cases}$$

### 4. Acil Kaçınma Direksiyonu (AES) Tetikleme Kuralı

$$\text{AES Aktif} \iff \left( d_{\text{obstacle}} < 0.75 \cdot d_{\text{stop}} \right) \land \left( \text{Yan Şerit Boş} \right)$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Sürücü veya otopilot anlık dikkatsizlik yaşasa bile öndeki duran araca veya yayaya çarpmayı önlemek / çarpışma enerjisini minimuma indirmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Arkadan Çarpma Kazaları:** Otoyol ve şehir içi zincirleme kazaları %80 oranında engelledi.
- **Yetersiz Fren Mesafesinde Kaçış:** Fren mesafesinin yetmediği anlarda otomatik kaçınma direksiyonu (AES) ile kazadan kurtulma sağladı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Yanlış Pozitif Frenleme (Phantom Braking):** Gölgeleri veya üst geçitleri engel sanıp ani fren yapma riski (Vision-only çoklu kamera füzyonu ile filtrelenir).
- **Arkadaki Trafik:** Çok sert frenleme sırasında arkadan gelen dikkatsiz araçların çarpma riskini minimize etmek için optimum frenleme uygulanır.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Yalnızca Sürücü Uyarısı (FCW):** Sürücünün tepki veremediği panik anlarında kazayı önleyemez.
- **Yavaş Kademeli Fren:** Yüksek hızlarda kinetik enerjiyi zamanında yok edemez.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **AEB (Autonomous Emergency Braking)** | Çarpışma anında otomatik olarak maksimum hidrolik fren basıncı uygulayan aktif güvenlik sistemi. |
| **FCW (Forward Collision Warning)** | Sürücüyü görsel ve sesli olarak uyaran çarpışma ön uyarı sistemi. |
| **AES (Autonomous Emergency Steering)**| Yalnızca frenlemenin yetmediği acil durumlarda aracı güvenli şeride yönlendiren kaçınma direksiyonu. |
| **Stopping Distance ($d_{\text{stop}}$)** | Sistemin tepki gecikmesi ile fiziksel fren mesafesinin toplamı. |
| **Hydraulic Delay ($t_{\text{delay}}$)** | Fren pedal komutunun hidrolik kaliperlere iletilip basınç oluşturması için geçen süre (~200 ms). |
| **Deceleration ($a$)** | Aracın birim zamandaki hız kaybı (Acil durumda Model 3: $-9.0\text{ m/s}^2 \approx 0.92g$). |
| **Euro-NCAP** | Avrupa Yeni Otomobil Değerlendirme Programı çarpışma ve güvenlik test standartları. |
| **Relative Speed ($v_{\text{rel}}$)** | Ego araç ile öndeki hedef araç arasındaki hız farkı ($v_{\text{ego}} - v_{\text{target}}$). |
| **Phantom Braking** | Yol üstünde gerçek bir engel yokken sistemin yanlışlıkla tam fren yapması arızası. |
| **iBooster** | Tesla araçlarında pedal hidroliğini milisaniyeler içinde elektronik olarak basınçlandıran fren aktüatörü. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Euro-NCAP 5-Yıldız güvenlik standartlarıyla tam uyum| • Phantom braking (hayalet fren) risk yönetimi şart   |
| • -9.0 m/s² (0.92g) maksimum durdurma kuvveti         | • Islak ve karlı zeminde durma mesafesinin uzaması    |
| • 5 µs ultra hızlı RTOS acil karar çevrimi            |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Gelecekteki Robotaksi filolarında sıfır ölümlü kaza | • Arkadaki aracın çok yakın takip etmesi ve           |
|   hedefine (Vision Zero) ulaşma                       |   AEB anında arkadan çarpma riski                     |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla AEB ve AES Karar Akış Şeması

```
[ Algılanan Engel Mesafesi (d_obs) ve Bağıl Hız (v_rel) ]
                           |
                           v
        [ TTC Hesabı ve Durma Mesafesi d_stop(v) ]
                           |
            +--------------+--------------+
            |                             |
            v                             v
[ d_obs < 0.75*d_stop VE Yan Şerit Açık ]  [ TTC <= 1.0s VEYA d_obs <= d_stop ]
            |                             |
            v                             v
[ ACİL KAÇINMA DİREKSİYONU (AES) ]     [ TAM ACİL FRENLEME (FULL AEB) ]
- Direksiyon: +0.40 rad                - İvme: -9.0 m/s² (0.92g)
- İvme: -4.0 m/s²                      - iBooster Maksimum Basınç
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana AEB simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
