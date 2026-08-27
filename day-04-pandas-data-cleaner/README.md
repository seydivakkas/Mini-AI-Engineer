# Day 04: Pandas Tabüler Veri Temizleme ve Ön İşleme Hatları

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.2+-150458.svg?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; bilgisayarlı görü ve yapay zeka sistemlerinde kullanılan görsel metaverilerini (çözünürlük, en-boy oranı, renk istatistikleri, etiketler) model eğitimine hazır hale getirmek için **eksik veri tamamlama (imputation)**, **IQR tabanlı aykırı değer sınırlama (clipping)**, **mükerrer kayıt eleme** ve **downcasting ile %50+ bellek tasarrufu** sağlayan üretim seviyesinde bir Pandas temizleme boru hattı sunar.

---

## 📌 Proje Kapsamı ve Mimari Genel Bakış

Makine öğrenimi modellerinin başarısı "Çöp İçeri, Çöp Dışarı" (Garbage In, Garbage Out) ilkesine bağlıdır. Kirli veriler modeli zehirler, eğitim sırasında kayıp fonksiyonunun (loss) ıraksamasına veya yanıltıcı yüksek test başarımlarına neden olur.

```
       Kirli Ham Tablo (NaN, Outliers, Duplicates, 64-bit Tipler)
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │ TabulerVeriTemizleyici │
                     │   (Fit - Transform)    │
                     └───────────┬────────────┘
                                 │
       ┌────────────────┬────────┴────────┬────────────────┐
       ▼                ▼                 ▼                ▼
[Mükerrer Eleme] [Eksik Tamamlama] [IQR Kırpma]   [Bellek Optimizasyonu]
Satır Duplikasyon Medyan / Mod     [Q1-1.5*IQR,   Downcasting: int8/32,
Temizliği         İmputasyonu      Q3+1.5*IQR]    float32, category
```

---

## 🧮 Matematiksel ve İstatistiksel Yöntemler

### 1. Veri Sızıntısı (Data Leakage) Koruması
Tüm veri kümesinin ortalaması/medyanı ile eksik değer doldurmak en yaygın veri sızıntısıdır. Temizleyici Scikit-Learn benzeri **`fit-transform`** mimarisi kullanır:
- **`fit()`:** Parametreler (medyan, mod, IQR çeyreklikleri) **YALNIZCA** eğitim kümesinden öğrenilir.
- **`transform()`:** Öğrenilen bu parametreler test veya canlı üretim verisine uygulanır.

### 2. Çeyrekler Açıklığı (IQR) ile Aykırı Değer Kırpma (Winsorization)
Verinin %25'lik ($Q_1$) ve %75'lik ($Q_3$) çeyreklikleri arasındaki fark $IQR = Q_3 - Q_1$ olarak tanımlanır:

$$\text{Alt Sınır} = Q_1 - 1.5 \cdot IQR, \quad \text{Üst Sınır} = Q_3 + 1.5 \cdot IQR$$

Veriler silinmek yerine bu sınırlara kırpılarak (clipping) bilgi kaybı engellenir ve aşırı uç noktaların model ağırlıklarını bozması önlenir.

### 3. Bellek Optimizasyonu (Downcasting)
- `int64` ($8$ bayt) $\to$ `int8` ($1$ bayt, $[-128, 127]$) veya `int16/int32`.
- `float64` ($8$ bayt) $\to$ `float32` ($4$ bayt).
- Düşük kardinaliteli metinler $\to$ `category` (Python string pointer yükünü kaldırır).

---

## 📂 Dizin Yapısı

```
day-04-pandas-data-cleaner/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar (pandas, numpy, pytest)
├── ana_akis.py                 # Konsol laboratuvar akışı
├── src/
│   ├── __init__.py
│   ├── veri_temizleyici.py     # TabulerVeriTemizleyici sınıfı
│   └── sentetik_veri_ureticisi.py # Kirli sentetik veri üreteci
└── testler/
    └── test_temizleyici.py     # 7 adet pytest birim testi
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleme
```bash
pip install -r gereksinimler.txt
```

### 2. Ana Akışı Çalıştırma
```bash
python ana_akis.py
```

### 3. Testleri Koşma
```bash
python -m pytest testler/test_temizleyici.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
