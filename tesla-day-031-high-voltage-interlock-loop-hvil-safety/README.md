# 🚗 Tesla Güvenlik Mimarisi | Gün 31: Yüksek Gerilim Kilidi (HVIL) ve Güvenlik Sistemleri

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Safety](https://img.shields.io/badge/ASIL-ASIL--D%20Functional%20Safety-red.svg?style=flat-square)](https://www.tesla.com/)
[![Standard](https://img.shields.io/badge/Standard-ISO%206469--1%20%2F%20UN%20ECE%20R100-blue.svg?style=flat-square)](https://www.sae.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Pyrofuse%20%3C2ms%20Actuation-orange.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"31. günümüze hoş geldin stajyer!  
> Bir elektrikli aracın tabanındaki $400\text{V}$ veya Cybertruck'taki $800\text{V}$ doğru akım gerilimi, insan kalbini durdurmaya yetecek seviyededir.  
> Bu yüzden bir Tesla'da yüksek gerilim güvenliği en katı otomotiv güvenlik seviyesi olan **ISO 26262 ASIL-D** standardıyla yönetilir:  
> 1. **HVIL (High Voltage Interlock Loop):** Tüm yüksek gerilim konnektörlerinden (İnvertör, Batarya Kapağı, DC Şarj Portu, Kompresör) geçen $88\text{ Hz}$ düşük gerilimli bir PWM güvenlik döngüsüdür. Serviste teknisyen bir fişi hafifçe oynatsa bile HVIL $5\text{ ms}$ içinde ana kontaktörleri açarak yüksek gerilimi keser.  
> 2. **Pyrofuse (Piroteknik Acil Durum Sigortası):** Kaza anında kontaktörler yapışabilir (Welded Contactor). Hava yastığı kontrolcüsü (RCM) çarpışma algıladığında batarya içindeki minik piroteknik fünyeyi $2\text{ ms}$ içinde patlatır ve yüksek gerilim barasını fiziksel olarak parçalayarak bataryayı izole eder.  
> 3. **Ön Şarj (Precharge) Sıralaması:** İnvertör girişindeki büyük DC link kapasitörlerini aniden $400\text{V}$'a bağlarsanız binlerce amperlik ark akımı kontaktörleri eritir. Önce bir ön şarj direnci üzerinden kapasitör $\%95$ doldurulur, ardından ana kontaktör kapatılır.  
> 4. **İzolasyon Direnci İzleme (ISO 6469-1):** Yüksek gerilim hatları ile araç şasisi arasındaki direnç sürekli ölçülür; $200\ \text{k}\Omega$ altına düşen herhangi bir gövde kaçağında araç acil moda geçer.  
> Bugün hayat kurtaran Tesla Yüksek Gerilim Güvenlik Çekirdeğini kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. İnvertör DC Link Ön Şarj (Precharge) Gerilim Denklemi
Batarya gerilimi $V_{\text{dc}}$, ön şarj direnci $R_{\text{pre}}$ ve invertör kapasitansı $C_{\text{link}}$ olmak üzere ($\tau = R_{\text{pre}} C_{\text{link}} \approx 80\text{ ms}$):

$$V_{\text{link}}(t) = V_{\text{dc}} \left(1 - \exp\left(-\frac{t}{\tau_{\text{precharge}}}\right)\right)$$

Ana kontaktör kapatma koşulu:

$$V_{\text{link}}(t) \ge 0.95 \cdot V_{\text{dc}}$$

### 2. İzolasyon Direnci Hesabı (ISO 6469-1 / UN ECE R100)

$$R_{\text{iso}} \ge 500\ \Omega/\text{V} \implies R_{\text{iso, min}} = 500 \times 400\text{ V} = 200\text{ k}\Omega$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Yüksek gerilimli elektrikli araçlarda ölümcül elektrik çarpması, ark patlaması, kaza sonrası yangın ve gövde kaçaklarını milisaniyeler içinde önlemek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Servis Güvenliği:** Bir turuncu yüksek gerilim kablosu sökülürken kontaklar ayrılmadan önce HVIL kesintiyi algılayıp gücü sıfırladı.
- **Kontaktör Kaynaması Engellendi:** Precharge devresi ile ana kontaktörlerin ani akımla yapışması (Welding) önlendi.
- **Kaza Anında Kesin İzolasyon:** Pyrofuse ile hava yastığı açıldığı an batarya paketi dış dünyadan fiziksel olarak koparıldı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Pyrofuse Geri Dönüşsüzdür:** Bir kere patlatıldığında batarya kapağı açılarak yeni piroteknik sigorta takılması gerekir.
- **Precharge Direnci Isınması:** Art arda kontak açıp kapatılırsa ön şarj direnci aşırı ısınabilir (Termal kilit gerekir).

### 4. Alternatifler Nelerdir? (Alternatives)
- **Geleneksel Ağır Eriyebilen Sigortalar:** Kaza anında yüksek akım çekilmezse erimez; Pyrofuse gibi aktif patlatılamaz.
- **Optik HVIL Döngüsü:** Fiber optik kablo kullanılır; maliyeti daha yüksektir.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **HVIL (High Voltage Interlock)** | Yüksek gerilim konnektörlerinin takılı olduğunu doğrulayan düşük gerilimli güvenlik döngüsü. |
| **Pyrofuse (Pyro-Switch)** | Kaza anında mikro patlayıcıyla devreyi $2\text{ ms}$ içinde fiziksel olarak kesen piroteknik sigorta. |
| **Precharge (Ön Şarj)** | İnvertör kapasitörlerini ana kontaktörler kapanmadan önce bir direnç üzerinden yumuşakça şarj etme işlemi. |
| **Main Contactors** | Bataryanın pozitif ve negatif kutuplarını DC baraya bağlayan yüksek akımlı elektromekanik röleler. |
| **Welded Contactor** | Aşırı akım arkı sebebiyle kontaktör kontaklarının birbirine kaynayarak yapışık kalması arızası. |
| **Isolation Resistance ($R_{\text{iso}}$)** | Yüksek gerilim hatları ile aracın $12\text{V}$ şasi topraklaması arasındaki elektriksel yalıtım direnci. |
| **ASIL-D** | ISO 26262 fonksiyonel güvenlik standardında can kaybı riskini önleyen en yüksek güvenlik seviyesi. |
| **RCM (Restraints Control Module)** | Hava yastıklarını ve Pyrofuse patlatma sinyalini üreten çarpışma kontrol beyni. |
| **DC Link Capacitor** | İnvertör girişindeki gerilim dalgalanmalarını filtreleyen büyük değerli yüksek voltaj kapasitörü. |
| **Shoot-Through Protection** | İnvertör köprüsünde kısa devre oluşmasını engelleyen donanımsal koruma. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • ASIL-D uyumlu < 2 ms Pyrofuse acil güç kesme        | • Pyrofuse patladığında serviste fiziksel değişim     |
| • 88 Hz PWM ile gürültüye dayanıklı HVIL döngüsü      | • Precharge direncinin tekrarlayan denemelerde ısısı  |
| • 0.85 µs ultra hızlı 1 kHz güvenlik karar çevrimi    |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Katı hal (Solid-State) e-Fuse teknolojisi ile       | • Nem ve korozyon sebebiyle izolasyon direncinin      |
|   yeniden kurulabilir mikro-patlatmasız koruma        |   zamanla düşerek sahte arıza üretmesi                |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Sistem Mimarisi & Yüksek Gerilim Güvenlik Döngüsü

```
             [ RCM Kaza Beyni ] ------(2 ms Acil Sinyal)------> [ Pyrofuse Fünye ]
                                                                       |
                                                                       v
   [ 400V Batarya ] ====[ Pyrofuse ]====[ Pozitif Kontaktör ]===[ DC Bara ]===> [ İnvertör ]
                                   |                                |
                                   +---[ Precharge Direnci ]--------+
                                   |
   [ 400V Batarya ] ====================[ Negatif Kontaktör ]==================> [ İnvertör ]

             [ 88 Hz PWM HVIL Döngüsü: Tüm HV Konnektörlerden Geçer ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana yüksek gerilim güvenlik simülasyon akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
