# 🚗 Tesla Güç Aktarma ve Frenleme | Gün 30: Rejeneratif Frenleme ve Enerji Geri Kazanım Algoritmaları

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Regen](https://img.shields.io/badge/Braking-One--Pedal%20Drive%20Hold%20Mode-blue.svg?style=flat-square)](https://www.tesla.com/)
[![Energy](https://img.shields.io/badge/Efficiency-Up%20to%2075kW%20Regen%20Power-green.svg?style=flat-square)](https://www.sae.org/)
[![Safety](https://img.shields.io/badge/Protection-SOP%20Cold%20%26%20Full%20SoC%20Limit-orange.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"30. günümüze hoş geldin stajyer!  
> Geleneksel içten yanmalı motorlu bir araçta fren pedalına bastığınızda aracın tonlarca ağırlığındaki kinetik enerjisi disk ve balatalarda sürtünme ısısına dönüşüp havaya uçar.  
> Bir Tesla'da ise gaz pedalından ayağınızı çektiğiniz an çekiş motoru anında bir **Jeneratöre** dönüşür ve aracı yavaşlatırken bataryaya $75\text{ kW}$'a varan elektrik basar (**Tek Pedallı Sürüş - One-Pedal Drive**)!  
> 1. **Tork Harmanlama (Braking Torque Blending):** Sürücü fren pedalına bastığında yazılım önce motorun maksimum rejenerasyon torkunu ($300\text{ Nm}$) kullanır. Sadece ani ve sert panik duruşlarında açığı hidrolik sürtünme balatalarıyla tamamlar.  
> 2. **SOP Şarj Kabul Kısıtlamaları:**  
>    - Soğuk Batarya ($T < 0^\circ\text{C}$): Dondurucu soğukta lityum iyonları anota giremez ve grafit üzerinde metalik lityum kaplama (Plating) yaparak bataryayı patlatabilir. Bu yüzden rejen gücü $0\text{ kW}$'a kısılır (Ekranda noktalı çizgi görünür).  
>    - $\%100$ Dolu Batarya: Aşırı voltajı önlemek için rejen kapatılır.  
> 3. **Hold Modu (0 km/h Duruş):** Araç $0\text{ km/h}$ hıza ulaştığında motor ters akım basarak ve elektronik park frenini devreye sokarak yokuşta dahi sıfır kayma ile sabit kalır.  
> Bugün balata ömrünü 150,000 km'ye çıkaran ve menzili $\%20+$ artıran rejenerasyon kontrolcüsünü inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Tork Harmanlama (Braking Torque Blending)

$$T_{\text{demanded, total}} = T_{\text{regen, requested}} + T_{\text{brake\_pedal}}$$

$$T_{\text{regen, actual}} = \min\left(T_{\text{demanded, total}}, T_{\text{regen, max}} \cdot \text{SOP}_{\text{factor}}\right)$$

$$T_{\text{hydraulic, actual}} = \max\left(0, T_{\text{demanded, total}} - T_{\text{regen, actual}}\right)$$

### 2. Geri Kazanılan Elektriksel Güç (Regenerative Electric Power)
Tekerlek yarıçapı $r_{\text{wheel}}$, araç hızı $v$ ($\text{m/s}$) ve sistem verimi $\eta \approx 0.90$ olmak üzere:

$$\omega_{\text{wheel}} = \frac{v}{r_{\text{wheel}}}$$

$$P_{\text{regen}} = T_{\text{regen, actual}} \cdot \omega_{\text{wheel}} \cdot \eta_{\text{inverter\_motor}}$$

$$E_{\text{recovered}} = \int_{t_0}^{t_{\text{stop}}} P_{\text{regen}}(t) \, dt$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Şehir içi dur-kalk trafiğinde kaybedilen kinetik enerjiyi geri kazanarak menzili $\%15-25$ artırmak ve mekanik fren balata aşınmasını yok etmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Tek Pedallı Konfor:** Sürücünün trafikte sürekli gaz ve fren arasında ayak değiştirmesini önledi; gazı bıraktığında $0.25\text{g}$ lineer duruş sağladı.
- **Balata Ömrünü Uzattı:** Disk frenlerin kullanımını $\%90$ azaltarak balata değişim ihtiyacını 150,000+ kilometreye taşıdı.
- **Hücre Koruma (SOP):** Soğuk havalarda lityum kaplama (Plating) riskine karşı rejeneratif şarj akımını otomatik kısıtladı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Düşük Hız Dinamiği ($< 3\text{ km/h}$):** Motor devri sıfıra yaklaştığında üretilen elektromotor kuvveti (BEMF) düşer; $0\text{ km/h}$ duruş için aktif tork enjeksiyonu gerekir.
- **Kaygan Zemin / ESP Müdahalesi:** Kar veya buzda arkadan itişli rejen aracın arkasını kaydırabilir; ESP devreye girdiği anda rejen derhal sıfırlanmalıdır.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Serbest Süzülme (Coast / Roll Modu):** Gaz bırakıldığında araç boş vitesteymiş gibi süzülür; rejen yalnızca fren pedalına basıldığında devreye girer.
- **Mekanik Sürtünme Freni:** Basit ancak tüm enerjiyi ısı olarak israf eder.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Regenerative Braking** | Elektrik motorunu jeneratör modunda çalıştırarak kinetik enerjiyi elektriğe dönüştürme işlemi. |
| **One-Pedal Drive** | Sadece gaz pedalına basıp çekerek aracın ivmelenmesini, yavaşlamasını ve tam durmasını sağlayan sürüş modu. |
| **Torque Blending** | Rejeneratif motor freni ile hidrolik sürtünme balatalarının gücünü sarsıntısız birleştiren algoritma. |
| **SOP (State of Power)** | Bataryanın o anki sıcaklık ve doluluğuna göre izin verilen maksimum anlık şarj gücü sınırı. |
| **Hold Mode** | Araç $0\text{ km/h}$'ye ulaştığında ayağı pedallara basmadan aracı yokuşta dahi sabit tutan duruş kilidi. |
| **Creep Mode** | Gaz bırakıldığında klasik otomatik vitesli araçlar gibi $5\text{ km/h}$ hızla yavaşça ilerleme modu. |
| **Lithium Plating** | Soğuk bataryaya yüksek şarj akımı basıldığında lityumun katı metal olarak çöküp kısa devre yapma riski. |
| **Deceleration Rate** | Gaz pedalının bırakılmasıyla elde edilen doğal frenleme ivmesi ($0.15\text{g} - 0.25\text{g}$). |
| **Back-EMF (Zıt EMF)** | Motor dönerken sargılarında endüklenen ve rejenerasyonu mümkün kılan gerilim. |
| **Friction Brake Wear** | Hidrolik fren balatalarının mekanik sürtünmeyle aşınması ve toz üretmesi. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Şehir içinde %20+ menzil artışı ve 112 Wh/duruş     | • -5°C soğukta veya %100 SoC'de rejenin kısılması     |
| • Fren balatası aşınmasında %90 radikal azalma        | • Kaygan zeminde rejenin arkadan kayma riski          |
| • 0.95 µs gerçek zamanlı tork harmanlama kararı       |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • ABS ve Çekiş Kontrolü (TC) ile milisaniyelik entegre| • Uzun süre kullanılmayan disk frenlerin paslanması   |
|   rejeneratif kayma önleme stabilitesi                |   ve acil durumlarda tutuş kaybı riski                |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Sistem Mimarisi & Tork Harmanlama Akış Şeması

```
     Sürücü Girdisi:
     - Gaz Pedalı (%0 Bırakıldı) ----+
     - Fren Pedalı (%30 Basıldı) ----|
                                     v
                  +-----------------------------------+
                  |  Tork Harmanlama Yöneticisi       |
                  |  - Talep: T_total = T_reg + T_hyd |
                  |  - SOP Kısıtı: Temp & SoC Limit   |
                  +-----------------+-----------------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
         [ Rejeneratif Motor Freni ]      [ Hidrolik Sürtünme Freni ]
         (Öncelikli: Max 300 Nm)          (Kalan Açık: Balatalar)
                    |                               |
                    +---------------+---------------+
                                    |
                                    v
                  [ Net Araç Yavaşlaması: 0.25g ]
                  [ 0 km/h Hold Modu Duruş Kilidi ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana rejeneratif frenleme akışını ve görselleştirme panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
