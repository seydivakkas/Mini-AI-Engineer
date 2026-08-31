# 🚗 Tesla FSD Otonom Sürüş | Gün 80: Dağıtık Güç Dağıtımı ve Dinamik Şebeke Yük Dengeleme Algoritmaları

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Load Balancing](https://img.shields.io/badge/Grid-Dynamic%20Load%20Balancer-red.svg?style=flat-square)](https://www.tesla.com/)
[![Megawatt](https://img.shields.io/badge/Capacity-1.0%20MW%20Grid%20Guard-blue.svg?style=flat-square)](https://en.wikipedia.org/wiki/Tesla_Supercharger)
[![Safety](https://img.shields.io/badge/Protection-Zero%20Transformer%20Overload-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"80. günümüze hoş geldin stajyer!  
> Bir Supercharger istasyonunda 8 veya 16 araç aynı anda şarj olurken ne olur?  
> Eğer her araç aynı anda $250\text{ kW}$ çekmeye çalışırsa, toplam talep $8 \times 250 = 2000\text{ kW}$ ($2.0\text{ MW}$) olur. Ancak yerel elektrik trafosunun kapasitesi yalnızca $1.0\text{ MW}$ ($1000\text{ kW}$) olabilir!  
> Trafonun patlamasını veya şebeke sigortasının atmasını engellemek için Tesla **Dinamik Şebeke Yük Dengeleme (Dynamic Load Balancing) ve Dağıtık Güç Paylaşımı** algoritmasını kullanır:  
> 1. **SoC Tabanlı Önceliklendirme:** Bataryası boş olan (%10 SoC) araç yüksek güç talep ederken, bataryası neredeyse dolu olan (%85 SoC) araç daha az güç alabilir ($D_i = 100 - \text{SoC}_i$).  
> 2. **Adil ve Optimum Paylaşım:** Mevcut $1.0\text{ MW}$ trafo gücü araçların ihtiyaçlarına göre milisaniyeler içinde paylaştırılır.  
> 3. **Artık Güç Yeniden Dağıtımı:** Stall limitine ($250\text{ kW}$) ulaşmış araçlardan artan güç diğer araçlara aktarılır.  
> 4. **Sıfır Trafo Aşımı:** Toplam çekilen güç asla trafo sınırını geçmez ($\sum P_i \le 1000\text{ kW}$).  
> Bugün binlerce Supercharger istasyonunun çökmeden tam kapasite çalışmasını sağlayan yük dengeleme motorunu inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Araç Güç Talebi Ağırlık Fonksiyonu

$$D_i = \max\left(1.0, \ 100.0 - \text{SoC}_i\right)$$

### 2. Dinamik Güç Dağıtımı ve Kırpma (Clamping) Kanunu

$$P_{i,\text{initial}} = \frac{D_i}{\sum_{j=1}^N D_j} \cdot P_{\text{grid\_capacity}}$$

$$P_i = \min\left( P_{\text{max\_stall}}, \ P_{i,\text{initial}} + \Delta P_{\text{residual},i} \right)$$

### 3. Trafo Aşırı Yük Koruması Değişmezi (Invariant)

$$\sum_{i=1}^N P_i \le P_{\text{grid\_capacity}} = 1000.0\text{ kW}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Sınırlı şebeke trafo altyapısına sahip alanlarda, trafo yükseltme maliyetine (milyonlarca dolar) girmeden çok sayıda aracın aynı anda şarj edilmesini sağlamak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Şebeke Çökmeleri ve Sigorta Atması:** İstasyon toplam güç çekişini anlık izleyip sınırlayarak trafo aşırı yüklenmelerini sıfıra indirdi.
- **Şarj Süresi Optimizasyonu:** Bataryası acil boş olan araçlara maksimum güç vererek istasyon kuyruk sürelerini kısalttı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Tüm Araçlar Boş İse:** İstasyon tamamen boş bataryalı 8 araçla dolarsa, her araca düşen güç $125\text{ kW}$ ile sınırlanır (Megapack desteği gerektirir).

### 4. Alternatifler Nelerdir? (Alternatives)
- **Sabit Güç Bölme (Statik Paylaşım):** Her stall'a sabit $125\text{ kW}$ vermek (Dolu araçların boşa ayrılan kapasitesini israf eder).
- **İlk Gelen İlk Alır (FIFO):** Sonradan gelen aracı tamamen bekletmek (Kullanıcı deneyimi için çok kötüdür).

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Dynamic Load Balancing** | Mevcut elektrik gücünün değişken tüketici taleplerine göre gerçek zamanlı paylaştırılması. |
| **Grid Headroom** | Trafo maksimum kapasitesi ile anlık kullanılan güç arasındaki kullanılabilir emniyet marjı. |
| **State of Charge (SoC)** | Bataryanın mevcut doluluk oranı (%0 ile %100 arası). |
| **Transformer Rating** | Bir elektrik trafosunun güvenle sağlayabileceği sürekli maksimum güç kapasitesi (kVA / kW). |
| **Power Curtailment** | Şebeke sınırını aşmamak için belirli tüketicilerin çektiği gücün kasıtlı kısılması. |
| **Residual Power** | Stall sınırına ulaşan araçlardan sonra boşta kalan ve diğerlerine dağıtılan artık güç. |
| **Peak Shaving** | İstasyonun pik saatlerde şebekeden aşırı güç çekmesini bataryalarla engelleme yöntemi. |
| **Droop Allocation** | Şebeke yüküne göre güç paylaştıran dağıtık kontrol algoritması. |
| **Stall** | Bir Supercharger istasyonunda tek bir aracın yanaşıp şarj olduğu park alanı / şarj direği. |
| **Brownout Prevention** | Aşırı yüklenme sonucu şebeke geriliminin çökmesini önleyen yazılımsal koruma. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %100 trafo aşım koruması (Sıfır şebeke çökmesi)     | • İstasyon tamamen boş araçlarla doluysa araç başına  |
| • SoC ters orantılı adil ve hızlı şarj dağıtımı       |   şarj hızının düşmesi                                |
| • 3 µs ultra hızlı RTOS optimizasyon döngüsü          |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Megapack batarya tamponu eklenerek 1 MW trafo ile   | • İstasyon içi CAN/Ethernet haberleşme kopukluğunda   |
|   3 MW anlık şarj gücü sunulabilmesi                  |   güvenli düşük güç moduna geçiş zorunluluğu          |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Supercharger Dinamik Yük Dengeleme Şeması

```
[ 1.0 MW (1000 kW) Şebeke Trafosu ]
                 |
                 v
   [ Tesla Master Site Controller ]
                 |
                 | 1. Anlık SoC Talepleri Okunur: D_i = 100 - SoC_i
                 v
   [ Dinamik Yük Dağıtım Algoritması ]
                 |
    +------------+------------+------------+
    |                         |            |
Stall 1 (%10 SoC)        Stall 4 (%55 SoC) Stall 8 (%92 SoC)
-> 240 kW (Max Hız)      -> 110 kW        -> 45 kW (Düşük Akım)
    |                         |            |
    +------------+------------+------------+
                 v
[ Toplam Güç: 960 kW <= 1000 kW (TRAFO %100 GÜVENDE) ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Yük Dengeleme simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
