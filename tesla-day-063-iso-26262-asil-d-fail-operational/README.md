# 🚗 Tesla FSD Otonom Sürüş | Gün 63: ISO 26262 Fonksiyonel Güvenlik (ASIL-D) ve Arıza Güvenli (Fail-Operational) Mimari

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![ASIL-D](https://img.shields.io/badge/Standard-ISO%2026262%20ASIL--D-red.svg?style=flat-square)](https://www.tesla.com/)
[![Fail-Operational](https://img.shields.io/badge/Architecture-Fail--Operational%20MRM-blue.svg?style=flat-square)](https://www.sae.org/)
[![Redundancy](https://img.shields.io/badge/Safety-Dual--Channel%20Cross--Check-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"63. günümüze hoş geldin stajyer!  
> Bir bilgisayar donarsa yeniden başlatırsınız. Ancak $130\text{ km/h}$ hızla otoyolda giden bir otonom aracın direksiyon kontrolcüsü çökerse sonuç felaket olur!  
> Otomotiv yazılımının en katı güvenlik standardı **ISO 26262 ASIL-D (Automotive Safety Integrity Level - D)**'dir:  
> 1. **ASIL Seviyelendirmesi (Severity, Exposure, Controllability):** Direksiyon ve fren kontrolü insan hayatını doğrudan etkilediği için en üst düzey olan ASIL-D sınıfındadır.  
> 2. **Çift Kanallı Çapraz Doğrulama (Dual-Channel Cross-Check):** Direksiyon tork sensörleri ve tekerlek hız sensörleri iki bağımsız hattan ($S_1, S_2$) okunur. Fark $\Delta S > 0.5\text{ Nm}$ olursa donanım hatası algılanır.  
> 3. **Arıza Filtreleme (Debounce Monitor):** Anlık elektriksel gürültüleri filtrelemek için 3 ardışık çevrim boyunca arıza devam ederse ASIL-D arıza bayrağı tetiklenir.  
> 4. **Fail-Operational ve Minimal Risk Manevrası (MRM):** Sistem arıza anında aniden kapanmaz (Fail-Silent/Fail-Safe yetersizdir); aracı kontrollü bir şekilde $-1.5\text{ m/s}^2$ ivmeyle emniyet şeridine çekip güvenle durdurur (Fail-Operational).  
> Bugün Tesla'nın sıfır kaza felsefesini donanım güvencesine bağlayan ASIL-D kalkanını inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Çift Kanallı Sensör Çapraz Doğrulama Kuralı

$$|S_{\text{ch1}}(t) - S_{\text{ch2}}(t)| \le \epsilon_{\text{threshold}}, \quad \epsilon_{\text{torque}} = 0.50\text{ Nm}, \ \epsilon_{\text{speed}} = 0.40\text{ m/s}$$

### 2. Debounce Arıza Sayacı Akış Modeli

$$C(t) = \begin{cases} C(t-1) + 1, & \text{Eğer } |S_1 - S_2| > \epsilon \\ \max(0, C(t-1) - 1), & \text{Eğer } |S_1 - S_2| \le \epsilon \end{cases}$$

$$\text{ASIL\_D\_FAULT} = \text{TRUE} \iff C(t) \ge N_{\text{threshold}} \quad (N = 3)$$

### 3. Fail-Operational Minimal Risk Manevrası (MRM)

$$a_{\text{mrm}} = -1.50\text{ m/s}^2, \quad \dot{\psi}_{\text{mrm}} \to \text{Emniyet Şeridi Yörüngesi}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Tek bir sensör, kablo kopması veya mikroişlemci arızasında bile aracın kontrolden çıkmasını önleyip emniyetli duruş garantisi sağlamak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Tek Nokta Hataları (Single Point of Failure):** Bir kanal bozulduğunda diğer kanal üzerinden durumu anlayıp güvenli manevrayı başlattı.
- **Yanlış Alarm İptalleri:** Debounce filtresiyle geçici voltaj dalgalanmalarının sistemi kapatmasını engelledi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Ortak Nedenli Arızalar (Common Cause Failures):** Her iki kanalın da aynı anda etkilendiği güç kaynağı çöküşlerine karşı çift bağımsız akü hattı gerekir.
- **Maliyet ve Donanım Yükü:** Çift sensör ve çift aktüatör mimarisi maliyet ve kablolama ağırlığını artırır.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Fail-Safe / Fail-Silent (Geleneksel):** Arıza anında sistemi kapatıp kontrolü insana devreder (Sürücüsüz Robotaksi için KABUL EDİLEMEZ).
- **Üçlü Yedeklilik (Triple Modular Redundancy - TMR / 2-out-of-3 Voting):** Havacılıkta kullanılır; otomotiv için aşırı maliyetlidir.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **ISO 26262** | Karayolu taşıtları fonksiyonel güvenlik uluslararası mühendislik standardı. |
| **ASIL-D** | ISO 26262 kapsamındaki en katı ve en yüksek risk seviyesi (Ölümcül kaza önleme). |
| **Fail-Operational** | Birincil sistem arızalansa bile ikincil yedek sistemle çalışmayı sürdürme yeteneği. |
| **Fail-Safe** | Arıza anında sistemi güvenli bir şekilde kapatıp enerjisini kesme yaklaşımı. |
| **Minimal Risk Maneuver (MRM)**| Kritik arıza anında otonom aracın yol kenarına çekip durmasını sağlayan acil durum manevrası. |
| **Dual-Channel** | İki ayrı fiziksel hattan bağımsız veri toplayan yedekli sensör mimarisi. |
| **Fault Debouncing** | Geçici elektriksel gürültüler ile kalıcı donanım arızalarını ayıran zaman filtresi. |
| **HARA** | Tehlike Analizi ve Risk Değerlendirmesi (Hazard Analysis and Risk Assessment). |
| **FMEA** | Hata Türü ve Etkileri Analizi (Failure Mode and Effects Analysis). |
| **EPS (Electric Power Steering)** | Çift sargılı ve çift mikroişlemcili elektrik destekli direksiyon sistemi. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %100 ISO 26262 ASIL-D uluslararası sertifikasyon    | • Çift kanal donanım karmaşıklığı ve maliyeti        |
| • 2 µs ultra hızlı RTOS güvenlik döngüsü              | • Ortak nedenli güç kesintilerine karşı yedek         |
| • Sıfır kilitlenme ile Fail-Operational duruş         |   akü tasarımı zorunluluğu                            |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Seviye 4 / Seviye 5 direksiyonsuz Cybercab          | • Yüksek voltajlı elektromanyetik gürültülerin        |
|   filolarında yasal onay ve kamu güveni               |   (EMI) her iki kanala aynı anda sızması              |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla ASIL-D Güvenlik Kalkanı Akış Şeması

```
[ Çift Kanal Sinyaller: Tork (CH1, CH2) & Hız (CH1, CH2) ]
                            |
                            v
       [ Kanal Farkı Kontrolü: |S1 - S2| <= Epsilon ]
                            |
            +---------------+---------------+
            |                               |
            v                               v
[ Fark Eşik Altında: NOMINAL ]      [ Fark Eşik Üstünde: HATA ]
- Sürüşe Devam (%100 Onay)         - Hata Sayacı: C(t) = C(t-1) + 1
- Hata Sayacı Sıfırlanır                    |
                                            v
                            [ 3 Ardışık Çevrim Hata mı? ]
                                            |
                             +--------------+--------------+
                             |                             |
                             v                             v
                     [ Hayır: UYARI ]             [ Evet: ASIL-D ARIZA ]
                     - Filtreleniyor              - FAIL-OPERATIONAL MRM
                                                  - -1.5 m/s² Güvenli Duruş
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana ASIL-D Güvenlik simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
