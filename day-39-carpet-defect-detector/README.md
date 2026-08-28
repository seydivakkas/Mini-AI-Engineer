# Day 39: Halı Dokuma Hataları, Leke ve Kusur Tespiti (Carpet Defect Detector)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6.svg?style=flat-square&logo=scipy)](https://scipy.org/)
[![Pillow](https://img.shields.io/badge/Pillow-9.5+-005571.svg?style=flat-square)](https://python-pillow.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; jakarlı halı dokuma tezgahlarında ve konveyör bant kalite kontrol hatlarında ortaya çıkan **iplik kopmaları (çözgü/atkı kaçığı), yağ/boya damlamaları, dokuma delikleri ve iplik düğüm/topaklanma hatalarını** gerçek zamanlı olarak tespit eden, morfolojik filtrelerle gürültüden arındıran, geometrik kontur analiziyle sınıflandıran ve fabrika üretim hattı aksiyon protokolünü belirleyen endüstriyel bilgisayarla görme motorudur.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstriyel Dokuma Kalite Kontrolünün Önemi
* **Maliyet Tasarrufu:** Halı dokunurken fark edilmeyen bir çözgü kaçığı veya yağ lekesi, onlarca metrekarelik kumaşın hurdaya (scrap) ayrılmasına yol açar.
* **Erken Uyarı Mekanizması:** Kritik bir hata (delik/yırtık) anında dokuma tezgahının motorunu durdurmak (`HATTI_DURDUR`) makine kırılmalarını ve hammadde ziyanını engeller.
* **Zorluk:** Halı desenlerinin kendisi karmaşık renklere ve periyodik dokulara sahiptir. Hatanın bu desen varyasyonundan ayırt edilmesi gerekir.

```
                    ┌──────────────────────────────────────────────────────────┐
                    │          GİRİŞ TEST NUMUNESİ & REFERANS MODELİ           │
                    └────────────────────────────┬─────────────────────────────┘
                                                 │
                                                 ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │  1. KALINTI VE ANOMALİ HARİTASI ÇIKARIMI (Residual Map Extraction)           │
        │  - R(x,y) = |I_test(x,y) - I_ref(x,y)| (veya Yerel Gauss Arka Plan Modeli)   │
        │  - İstatistiksel Adaptif Eşikleme: T = mu + k * sigma                        │
        └────────────────────────────────────────┬─────────────────────────────────────┘
                                                 │
                                                 ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │  2. MORFOLOJİK FİLTRELEME & PARÇA BİRLEŞTİRME                                │
        │  - Açma (Opening = Erozyon -> Genişleme): Ayrık tekil doku gürültüsünü siler  │
        │  - Kapama (Closing = Genişleme -> Erozyon): Kopuk çizgi ve delikleri bağlar  │
        └────────────────────────────────────────┬─────────────────────────────────────┘
                                                 │
                                                 ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │  3. BAĞLANTILI BİLEŞENLER & KONTUR GEOMETRİSİ ANALİZİ                        │
        │  - Bounding Box [x, y, w, h] & Alan (Area A)                                 │
        │  - En-Boy Oranı (AR = max(w,h) / min(w,h))                                  │
        │  - Dairesellik / Kompaktlık: C = 4 * pi * A / (P^2)                          │
        └────────────────────────────────────────┬─────────────────────────────────────┘
                                                 │
                                                 ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │  4. ENDÜSTRİYEL KUSUR SINIFLANDIRMA & FABRİKA PROTOKOLÜ                      │
        │  - AR >= 3.2  -> IPLIK_KOPMASI (Orta Kusur -> 2. Kalite)                     │
        │  - C >= 0.45  -> YAG_BOYA_LEKESI (Orta Kusur -> Leke Çıkarma / 2. Kalite)    │
        │  - Alan >= 400 -> DELIK_YIRTIK (Kritik -> Hattı Durdur / Hurda)              │
        └──────────────────────────────────────────────────────────────────────────────┘
```

---

#

---

### 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Halı üretim hatlarında dokuma hatalarını, iplik kaçıklarını ve renk lekelerini morfolojik analiz ve derin öğrenme ile anında tespit etmek için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- İnsan denetçilerin göz yorgunluğuna bağlı kaçırdığı mikro üretim kusurlarını 7/24 kesintisiz ve standart kalitede yakalar.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Sentetik veya nadir görülen yeni kusur tiplerinde yeterli eğitim verisi yoksa false-negative oranı artabilir.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- Autoencoder / Anomaly Detection (PatchCore, Padim), YOLOv8-Defect veya Segmentasyon modelleri.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Görsel Kusur Tespiti** | *Visual Defect Inspection* | Halı dokuma yüzeyindeki ilmek kaçığı, leke, renk sapması ve yırtık gibi üretim anomalilerini tespit etme. |
| **Morfolojik Kusur Filtreleme** | *Morphological Defect Filtering* | Normal halı doku gürültüsünü filtreleyip yapısal süreksizlik gösteren anomali bölgelerini izole etme. |
| **Bölge Özellikleri (Regionprops)** | *Region Property Analysis* | Tespit edilen kusur konturlarının alan, eksen uzunluğu ve dairesellik metrikleriyle şiddetini derecelendirme. |
| **Kusur Isı Haritası** | *Defect Anomaly Heatmap* | Kusurun yoğunlaştığı koordinatları operatöre kırmızı uyarı bölgeleri olarak gösteren maske. |

---

## 2. Matematiksel Formülasyonlar

#### A. Morfolojik Operatörler
* **Erozyon ($A \ominus B$):** $A \ominus B = \{z \mid (B)_z \subseteq A\}$
* **Genişleme ($A \oplus B$):** $A \oplus B = \{z \mid (\hat{B})_z \cap A \neq \emptyset\}$
* **Açma ($A \circ B$):** $(A \ominus B) \oplus B$ (Gürültü yok etme)
* **Kapama ($A \bullet B$):** $(A \oplus B) \ominus B$ (Kusur sınırlarını ve iç boşlukları kapatma)

#### B. Kontur ve Şekil Geometrisi Metrikleri
* **En-Boy Oranı (Aspect Ratio):** $AR = \frac{\max(w, h)}{\min(w, h)}$ (İplik kaçıklarında $AR \ge 3.2$)
* **Dairesellik (Circularity):** $C = \frac{4\pi \cdot \text{Alan}}{\text{Çevre}^2}$ ($C \approx 1.0$ tam dairesel leke, $C \ll 1.0$ ince uzun çizgi)
* **Doluluk Oranı (Extent / Density):** $\text{Doluluk} = \frac{\text{Alan}}{w \times h}$

---

### 3. Kusur Tespit Deney Çıktıları

| Kusur ID | Kusur Türü | Alan (px) | En-Boy Oranı ($AR$) | Dairesellik ($C$) | Şiddet | Fabrika Üretim Aksiyonu |
|---|---|---|---|---|---|---|
| **DEFECT-01** | `IPLIK_KOPMASI` | **960 px** | **40.00** | **0.08** | `KRITIK` | `HATTI_DURDUR_HURDA_AYIR` |
| **DEFECT-02** | `YAG_BOYA_LEKESI`| **1520 px**| **1.00** | **0.86** | `KRITIK` | `HATTI_DURDUR_HURDA_AYIR` |
| **DEFECT-03** | `DUGUM_TOPAKLANMA`| **120 px** | **1.00** | **0.82** | `KUCUK_KUSUR`| `REWORK_DUZELTME_ISTASYONU` |

**Fabrika Kalite Kararı:** `PARTI_RED_HURDA` (Kritik dokuma ve leke hataları tespit edildiği için parti sevkiyatı durduruldu).

---

## 🛠️ Dizin Yapısı

```
day-39-carpet-defect-detector/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # numpy, scipy, pillow, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca anomali tespiti ve kalite raporu betiği
├── README.md                        # 220+ satır sektörel ve matematiksel dokümantasyon
├── src/
│   ├── __init__.py
│   ├── anomali_tespitci.py          # Kalıntı haritası ve istatistiksel eşikleme motoru
│   ├── morfolojik_filtre.py         # Açma, kapama ve gürültü temizleyici
│   ├── kontur_analizci.py           # Bağlantılı bileşenler, bounding box ve şekil analizi
│   ├── kusur_siniflandirici.py      # Kural tabanlı kusur sınıflandırma ve QC protokolü
│   ├── sentetik_kusur_uretici.py    # Dokuma hataları enjektörü ve sentetik veri üretici
│   └── gorsellestirici.py           # 6 panelli endüstriyel kalite kontrol panosu
├── testler/
│   ├── __init__.py
│   └── test_carpet_defect.py        # 7 adet birim test (Tümü Başarılı)
└── ciktilar/
    └── hali_kusur_tespit_paneli.png # 6 panelli yüksek çözünürlüklü teşhis görseli
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkların Kurulması
```bash
pip install -r gereksinimler.txt
```

### 2. Ana Akışın Çalıştırılması
```bash
python ana_akis.py
```

### 3. Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** `src/kusur_siniflandirici.py` içerisine halı üzerindeki **"Kusur Yoğunluk Haritası (Defect Density Grid)"** fonksiyonunu ekleyerek halıyı $3 \times 3$ grid bölgelerine bölmek ve hangi bölgede (Sol-Üst, Merkez, Sağ-Alt vb.) en çok kusur biriktiğini tespit etmek.

**Tamamlanan Çözüm:**
```python
def kusur_bolgesel_yogunluk(kusurlar: list, genislik: int = 400, yukseklik: int = 300) -> dict:
    grid_sonuc = {f"R{r+1}_C{c+1}": 0 for r in range(3) for c in range(3)}
    cell_w, cell_h = genislik / 3.0, yukseklik / 3.0
    for k in kusurlar:
        cx, cy = k["merkez"]
        c_idx = min(2, int(cx // cell_w))
        r_idx = min(2, int(cy // cell_h))
        grid_sonuc[f"R{r_idx+1}_C{c_idx+1}"] += 1
    return grid_sonuc
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Halı ve kumaş dokuma anomali tespitinde neden doğrudan basit piksel eşiklemesi (Simple Thresholding) kullanılamaz da **Morfolojik Açma (Opening) ve Kapama (Closing)** adımları zorunludur?

> **Mentor Cevabı:**
> Halı dokuma yüzeyleri doğal olarak iplik pürüzlülüğü, ışık yansıması ve mikro gölgelenmeler nedeniyle **yüksek frekanslı saçılmış gürültü pikselleri** üretir. Basit eşikleme yapıldığında binlerce tekil piksel hatalı kusur (false positive) olarak işaretlenir. 
> 1. **Açma (Opening):** Yapılandırıcı elemandan daha küçük olan bu tekil tuz-biber doku gürültülerini tamamen yok eder.
> 2. **Kapama (Closing):** İplik kopması gibi uzun kusurların dokuma aralıkları yüzünden parçalanmış ince çizgilerini tek bir sürekli kontur halinde köprüleyerek (bridging) birleştirir ve doğru geometrik sınıflandırma sağlar.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
