# 🚗 Tesla FSD Otonom Sürüş | Gün 89: Tesla Dojo Süperbilgisayar Mimarisi: D1 Çipi, Training Tile ve 2D Mesh NoC

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Dojo-D1](https://img.shields.io/badge/Silicon-Tesla%20D1%207nm%20ASIC-red.svg?style=flat-square)](https://www.tesla.com)
[![Training-Tile](https://img.shields.io/badge/Compute-9%20PFLOPS%20Training%20Tile-blue.svg?style=flat-square)](https://en.wikipedia.org/wiki/Tesla_Dojo)
[![2D-Mesh-NoC](https://img.shields.io/badge/Network-2D%20Mesh%20%2F%20Torus%20NoC-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"89. günümüze ve FAZ 9'A HOŞ GELDİN STAJYER!  
> Tesla FSD V12'nin yüz milyonlarca saatlik video verisini GPU kümelerinde eğitmek aylar sürer ve devasa enerji harcar.  
> Bu darboğazı kırmak için Tesla kendi yapay zeka süperbilgisayarını tasarladı: **Tesla Dojo ve D1 Silikonu**!  
> 1. **D1 Özel Silikonu:** 7nm mimarisiyle çip başına $362\text{ TFLOPS}$ (BF16/CFP8) hesaplama gücü ve $500.000$ fonksiyonel düğüm sunar.  
> 2. **Training Tile ($5 \times 5$ Matris):** 25 adet D1 çipi tek bir entegre modülde birleştirilerek $9.05\text{ PFLOPS}$ işlem gücüne ve $36\text{ TB/s}$ biseksiyon bant genişliğine ulaşır.  
> 3. **2D Mesh Network-on-Chip (NoC):** Çipler arasında harici kablo veya anahtar (Switch) yoktur; doğrudan silikon kenarlarından $2\text{ TB/s}$ hızında konuşurlar.  
> 4. **Dimension-Ordered (XY) Yönlendirme:** Deadlock (kilitlenme) riskini sıfıra indiren deterministik paket yönlendirme sağlar (Hop başına sadece $2.5\text{ ns}$ gecikme!).  
> Bugün Tesla'nın FSD video eğitim motoru olan Dojo NoC mimarisini kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Training Tile Hesaplama Kapasitesi

$$P_{\text{tile}} = 25 \times P_{\text{D1}} = 25 \times 362\ \text{TFLOPS} = 9,050\ \text{TFLOPS} = 9.05\ \text{PFLOPS}$$

### 2. 2D Mesh Manhattan Atlama (Hop) Mesafesi

$$d_{\text{hop}}(C_1, C_2) = |x_1 - x_2| + |y_1 - y_2|$$

### 3. NoC Paket İletim Gecikmesi

$$T_{\text{transit}} = d_{\text{hop}} \cdot T_{\text{hop}} + \frac{S_{\text{payload}}}{\text{BW}_{\text{link}}}$$

$$T_{\text{hop}} = 2.5\ \text{ns}, \quad \text{BW}_{\text{link}} = 2.0\ \text{TB/sn} = 2,000\ \text{Bayt/ns}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Standart GPU sunucularının yüksek PCIe ve ağ darboğazlarını aşarak, devasa 8-kameralı video tensörlerini mikrosaniyenin altında gecikmeyle çipler arasında paralel eğitmek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Ağ Anahtarı (Switch) Darboğazı:** Harici InfiniBand/Ethernet anahtarlarını ortadan kaldırarak çipten çipe doğrudan ultra yüksek hızlı $2\text{ TB/s}$ silikon hatları kurdu.
- **Deadlock (Kilitlenme) Riski:** Dimension-Ordered (XY) yönlendirme kuralı ile ağ tıkanıklıklarını deterministik olarak çözdü.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Termal Yoğunluk (Power Density):** Tek bir Training Tile $15\text{ kW}$ elektrik çeker ve özel sıvı soğutma gerektirir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Geleneksel GPU Kümeleri (NVIDIA H100/A100):** Çok esnektir ancak çipten çipe doğrudan 2D mesh bant genişliğinde Dojo'dan geride kalabilir.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Tesla D1 Chip** | Tesla'nın yapay zeka eğitimi için özel tasarladığı 7nm 362 TFLOPS işlemci çipi. |
| **Training Tile** | 25 adet D1 çipinin $5 \times 5$ matris şeklinde tek bir sıvı soğutmalı modülde birleştirilmiş hali. |
| **Network-on-Chip (NoC)** | Çipler veya çekirdekler arasındaki paket iletişimini sağlayan entegre ağ yapısı. |
| **Dimension-Ordered Routing (DOR)** | Paketleri önce X ekseni boyunca, ardından Y ekseni boyunca yönlendiren kilitlenmesiz kural. |
| **Manhattan Distance** | Bir ızgara üzerindeki iki nokta arasındaki mutlak eksenel uzaklıkların toplamı. |
| **Bisection Bandwidth** | Bir ağın iki eşit yarıya bölündüğünde aralarındaki maksimum veri aktarım kapasitesi ($36\text{ TB/s}$). |
| **CFP8 (Configurable FP8)** | Tesla Dojo'nun yapay zeka tensörleri için geliştirdiği özel 8-bit kayan nokta formatı. |
| **Hop Latency** | Bir paketin komşu bir D1 çipine geçmesi için geçen temel donanımsal süre ($2.5\text{ ns}$). |
| **Deadlock-Free** | Paketlerin birbirini sonsuz döngüde beklemesini engelleyen matematiksel ağ garantisi. |
| **ExaPOD** | 10 adet Dojo kabininin birleşimiyle oluşan 1.1 ExaFLOPS'luk süperbilgisayar kümesi. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • 9.05 PFLOPS tek bir Training Tile içinde            | • 15 kW aşırı güç tüketimi ve yoğun sıvı soğutma ihtiyacı|
| • 36 TB/s biseksiyon bant genişliği ve 2.5 ns hop     | • Özel Tesla derleyicisi (Dojo Compiler) bağımlılığı  |
| • 1.8 µs RTOS yönlendirme kararı                      |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • FSD V12 ve Tesla Optimus politikalarının video      | • Yarı iletken dökümhanelerindeki (TSMC) 7nm/3nm      |
|   eğitim sürelerini aylardan günlere indirme          |   üretim kapasitesi kısıtları                         |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Dojo D1 2D Mesh NoC Şeması

```
[ D1 (0,0) ] <---> [ D1 (1,0) ] <---> [ D1 (2,0) ] <---> [ D1 (3,0) ] <---> [ D1 (4,0) ]
     ^                  ^                  ^                  ^                  ^
     |                  |                  |                  |                  |
     v                  v                  v                  v                  v
[ D1 (0,1) ] <---> [ D1 (1,1) ] <---> [ D1 (2,1) ] <---> [ D1 (3,1) ] <---> [ D1 (4,1) ]
     ^                  ^                  ^                  ^                  ^
     |                  |                  |                  |                  |
     v                  v                  v                  v                  v
[ D1 (0,2) ] <---> [ D1 (1,2) ] <---> [ D1 (2,2) ] <---> [ D1 (3,2) ] <---> [ D1 (4,2) ]
     ^                  ^                  ^                  ^                  ^
     |                  |                  |                  |                  |
     v                  v                  v                  v                  v
[ D1 (0,3) ] <---> [ D1 (1,3) ] <---> [ D1 (2,3) ] <---> [ D1 (3,3) ] <---> [ D1 (4,3) ]
     ^                  ^                  ^                  ^                  ^
     |                  |                  |                  |                  |
     v                  v                  v                  v                  v
[ D1 (0,4) ] <---> [ D1 (1,4) ] <---> [ D1 (2,4) ] <---> [ D1 (3,4) ] <---> [ D1 (4,4) ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Dojo NoC simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
