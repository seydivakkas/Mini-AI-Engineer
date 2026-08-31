# 🚗 Tesla Termal Yönetim Sistemi | Gün 27: Isı Pompası ve Octovalve (8-Yollu Valf) Kontrol Algoritmaları

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Thermal](https://img.shields.io/badge/Thermal-Octovalve%208--Way%20Rotary%20Valve-blue.svg?style=flat-square)](https://www.tesla.com/)
[![Heat Pump](https://img.shields.io/badge/Heat%20Pump-COP%203.5%20Efficiency-green.svg?style=flat-square)](https://www.sae.org/)
[![Preconditioning](https://img.shields.io/badge/Supercharger-Battery%20Preconditioning%2045%C2%B0C-red.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"27. günümüze hoş geldin stajyer!  
> Bir elektrikli aracın kışın menzil kaybetmesinin en büyük sebebi kabin ve batarya ısıtması için harcanan enerjidir.  
> Eski elektrikli araçlar evdeki su ısıtıcıları gibi $1\text{ kW}$ elektrik harcayıp $1\text{ kW}$ ısı üreten dirençli PTC ısıtıcılar kullanıyordu ($COP = 1.0$).  
> Tesla Model Y ile hayatımıza giren **Octovalve (8-Yollu Döner Valf)** ve **Isı Pompası (Heat Pump)** mimarisi bu oyunu değiştirdi:  
> 1. **COP > 3.5 Verimi:** Isı pompası $1\text{ kW}$ elektrik harcayarak dış havadan veya motordan $3.5\text{ kW}$ termal ısı taşır ($\%70$'e varan kış menzili tasarrufu).  
> 2. **Octovalve 8-Yollu Dağıtıcı:** Tek bir step motorla 8 farklı hidrolik boru hattını 15 farklı termal modda birbirine bağlar (Gereksiz onlarca valfi ve hortumu ortadan kaldırır).  
> 3. **Supercharger Ön Isıtma (Preconditioning):** Navigasyonda Supercharger seçildiğinde, sürüş esnasında motorun ve invertörün kayıp ısısı toplanıp bataryaya basılır; batarya $45^\circ\text{C}$'ye getirilerek araca takıldığı an $250\text{ kW}$ pik şarj hızı elde edilir.  
> Bugün Tesla'nın termal mühendislik harikası olan Octovalve kontrol algoritmasını inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Kütlesel Isıl Diferansiyel Denklemi

$$C_{\text{th}} \frac{dT(t)}{dt} = \dot{Q}_{\text{in}}(t) - \dot{Q}_{\text{out}}(t)$$

$$T(t + \Delta t) = T(t) + \frac{\dot{Q}_{\text{net}} \cdot \Delta t}{C_{\text{th}}}$$

### 2. Isı Pompası Performans Katsayısı (COP)

$$\text{COP} = \frac{\dot{Q}_{\text{thermal}}}{W_{\text{electrical}}} = \frac{\dot{m} \cdot C_p \cdot (T_{\text{out}} - T_{\text{in}})}{P_{\text{compressor}}}$$

### 3. Batarya Ön Isıtma Toplam Isı Akısı (Preconditioning Heat Flux)

$$\dot{Q}_{\text{battery, net}} = (\text{COP} \cdot P_{\text{compressor}}) + \dot{Q}_{\text{powertrain, loss}} - h \cdot A \cdot (T_{\text{battery}} - T_{\text{ambient}})$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Kış aylarında dirençli ısıtıcıların sebep olduğu $\%30-40$ menzil kaybını önlemek ve bataryayı hızlı şarj için optimum $45^\circ\text{C}$ sıcaklığa getirmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Komponent Sayısını Azalttı:** Onlarca solenoid valf ve hortum yerine tek bir kompakt Octovalve gövdesi kullanıldı.
- **Motor Isı Hasadı:** İtme gücü motoru ve invertördeki $2.5\text{ kW}$ kayıp ısı boşa atılmayıp batarya ve kabine aktarıldı.
- **250 kW Supercharger Hazırlığı:** Soğuk havada bataryayı önceden ısıtarak şarj süresini 45 dakikadan 18 dakikaya indirdi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Aşırı Soğuk Limiti ($-20^\circ\text{C}$ Altı):** Dış havada çekilecek ısı kalmadığında motor rotoruna kasti reaktif akım basılarak (Stall heating) motor yapay ısıtıcı olarak kullanılır.
- **Mekanik Aşınma:** Döner valf contalarının aşınması hidrolik kaçaklara sebep olabilir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Dirençli PTC Isıtıcı:** Ucuzdur fakat $COP = 1.0$ olduğu için menzili ciddi şekilde düşürür.
- **Geleneksel Çift Döngülü Isıtma/Soğutma:** Karmaşık ve ağırdır; ısı geri kazanımı yetersizdir.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Octovalve** | 8 adet hidrolik portu tek bir döner göbekle birbirine bağlayan patentli 8-yollu döner valf. |
| **Heat Pump (Isı Pompası)** | Soğutucu akışkan faz değişimiyle düşük sıcaklıktan yüksek sıcaklığa ısı pompalayan termodinamik çevrim. |
| **COP (Coefficient of Performance)** | Harcanan elektrik gücüne karşılık üretilen/taşınan faydalı termal güç oranı ($COP > 3.0$). |
| **Preconditioning (Ön Isıtma)** | Supercharger şarjı öncesinde bataryayı en yüksek lityum iyon iletkenliği için $45^\circ\text{C}$'ye getirme işlemi. |
| **Chiller** | Batarya soğutma sıvısından ısı çekerek soğutucu gaz devresine aktaran plakalı ısı eşanjörü. |
| **PTC Heater** | Pozitif sıcaklık katsayılı elektrikli direnç ısıtıcısı ($COP = 1.0$). |
| **Thermal Mass ($C_{\text{th}}$)** | Bir bileşenin sıcaklığını $1\text{ K}$ artırmak için gereken ısı enerjisi ($J/K$). |
| **Powertrain Heat Harvesting** | Motor ve invertör kayıp ısısının soğutma sıvısıyla yakalanıp bataryaya yönlendirilmesi. |
| **Coolant Loop** | Su-Glikol karışımının batarya, motor ve radyatör arasında dolaştığı sıvı devresi. |
| **Defrost Mode** | Isı pompasının dış ünitedeki buzlanmayı çözmek için geçici olarak ters çevrimde çalışması. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %68.8 enerji tasarrufu ve COP 3.5 yüksek verim      | • -25°C aşırı dondurucu havalarda hava kaynaklı COP   |
| • Tek bir valf gövdesiyle 15 farklı termal mod        |   veriminin düşmesi                                   |
| • 0.95 µs gerçek zamanlı diferansiyel çözüm           | • Soğutucu gaz kaçaklarında tüm sistemin durması      |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Otonom navigasyonla entegre rota bazlı dinamik      | • Termal valf gövdesinde mekanik kirlilik veya        |
|   batarya sıcaklık ön planlaması                      |   partikül sıkışması riski                            |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Sistem Mimarisi & Octovalve Akış Şeması

```
                       +-------------------+
                       |    OCTOVALVE      |
                       |  (8-Yollu Valf)   |
                       +---------+---------+
                                 |
         +-----------------------+-----------------------+
         |                       |                       |
         v                       v                       v
  [ Batarya Paketi ]      [ Çift Motor & İnvertör ]  [ Kabin Isı Eşanjörü ]
  (450,000 J/K Isıl)      (2.5 kW Kayıp Isı)         (HVAC İklimlendirme)
         |                       |                       |
         +-----------------------+-----------------------+
                                 |
                                 v
                       [ Isı Pompası Kompresörü ]
                       (COP = 3.5 Enerji Verimi)
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana termal yönetim simülasyon akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
