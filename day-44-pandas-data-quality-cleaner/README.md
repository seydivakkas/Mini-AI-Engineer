# Day 44: Pandas ile Üretim Seviyesi Şema Doğrulama & Otomatik Veri Kalitesi Temizliği (Pandas Data Quality Cleaner)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0+-150458.svg?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** müfredatımızın 44. gününde geliştirilen **Üretim Seviyesi Tabüler Veri Şeması Doğrulama & Otomatik Veri Kalitesi Temizliği Motorudur**. Gerçek dünya veri boru hatlarında (ETL, Feature Store, Model Eğitimi, Tabüler API'ler) gelen kirli `pandas.DataFrame` tablolarını deklaratif kurallar (**Pandera & Great Expectations mantığı**) ile denetler, veri kalitesi skoru ($Q \in [0, 100]$) üretir ve otomatik **İmpütasyon**, **Sınır Kırpma (Clamping)** ve **Mükerrer Temizliği (Deduplication)** uygular.

---

## 📖 Mentorluk Dersi ve Veri Kalitesi Mühendisliği

### 1. Üretimde Kirli Verinin Tabüler Modellere Etkisi

Tabüler makine öğrenimi modellerinde (XGBoost, LightGBM, CatBoost, Scikit-Learn) model performansının %80'i veri hijyenine bağlıdır. Üretimde karşılaşılan 5 temel veri kalitesi boyutu şunlardır:

1. **Eksiklik (Completeness):** Sütunlardaki kontrolsüz `NaN` / `None` değerleri.
2. **Geçerlilik (Validity):** Sayısal sınırların aşılması (örn: `yaş < 0` veya `yaş > 120`, negatif fiyatlar).
3. **Format Uyumu (Consistency):** E-posta, telefon veya SKU kodlarının RegEx desenlerine uymaması.
4. **Tekillik (Uniqueness):** Birincil anahtarların (Primary Key / User ID) mükerrer olması veya yinelenen satırlar.
5. **Kategorik Bütünlük (Domain Conformity):** Önceden tanımlanmamış beklenmeyen metinlerin kategorik değişkenlere girmesi.

```
                           ┌──────────────────────────────────────────────────────────┐
                           │      HAM TABÜLER VERİ GİRDİSİ (Kirli pandas.DataFrame)   │
                           └────────────────────────────┬─────────────────────────────┘
                                                        │
                                                        ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      SemaDogrulayici (Deklaratif Tablo Sözleşmesi Denetimi)                       │
    │  - Tip & Kolon Kontrolü: Eksik zorunlu kolon, beklenmeyen fazla kolon tespiti                     │
    │  - Sayısal Sınırlar   : min_deger <= x <= max_deger (Aralık ihlalleri)                            │
    │  - Kategorik & RegEx  : İzinli kategori kümesi ve desen kontrolü                                  │
    │  - Eksik & Mükerrer   : Null oranları ve duplicate satır analizi                                  │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
    ┌──────────────────────────┐                                  ┌──────────────────────────┐
    │  KRİTİK RED (REJECT)     │                                  │  DÜZELTİLEBİLİR KİRLİ    │
    │  - Zorunlu Kolon Eksik   │                                  │  - Sınır Dışı Outlier    │
    │  - Benzersiz ID Duplicate│                                  │  - Eksik Null Değerler   │
    │  - Kalite Skoru < %60    │                                  │  - Mükerrer Satırlar     │
    └──────────────────────────┘                                  └────────────┬─────────────┘
                                                                               │
                                                                               ▼
                                                                  ┌──────────────────────────┐
                                                                  │  OtomatikVeriTemizleyici │
                                                                  │  - drop_duplicates()     │
                                                                  │  - clip(min, max)        │
                                                                  │  - fillna(median/mean)   │
                                                                  │  - Kategori Standartlama │
                                                                  └────────────┬─────────────┘
                                                                               │
                                                                               ▼
                                                                  ┌──────────────────────────┐
                                                                  │ ÜRETİME HAZIR TEMİZ VERİ │
                                                                  │ (Feature Store / Model)  │
                                                                  └──────────────────────────┘
```

---

#

---

### 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Tabüler veri kümelerindeki eksik değerleri, aykırı gözlemleri ve tip tutarsızlıklarını deterministik bir temizleme boru hattı ile dönüştürmek için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Makine öğrenimi modellerinin kirli veriler yüzünden düşük genelleme performansı göstermesini ve patlamasını engeller.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Çok büyük boyutlu (100GB+) büyük veri setlerinde tek makine bellek sınırlarına takılır.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- Polars, PySpark, Dask veya DuckDB.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Veri Kalitesi Kuralları** | *Data Quality Rule Engine* | Sütunların tip, değer aralığı, boşluk oranı ve desen uyumunu otomatik denetleyen doğrulama motoru. |
| **Winsorizasyon** | *Winsorization Truncation* | Aşırı uç aykırı değerleri silmek yerine belirli persentil sınırlarına (örn. %1 ve %99) çekerek veriyi koruma tekniği. |
| **Boru Hattı Eşkuvvetliliği** | *Pipeline Idempotency* | Aynı veri temizleme boru hattının birden fazla kez çalıştırıldığında veri üzerinde yan etki veya bozulma yaratmaması ilkesi. |
| **Şema Doğrulama** | *Tabular Schema Assertion* | Eğitim boru hattına giren tabloların sütun isimleri ve tiplerinin sözleşmeye uygunluğunu garanti etme. |

---

## 2. Kalite Skoru ve İmpütasyon Matematiksel Formülasyonları

- **Veri Kalitesi Skoru ($Q$):**
  $$Q = \max\left(0, 100 - \left(\frac{N_{\text{Hatalı Hücre}}}{N_{\text{Toplam Hücre}}} \times 100\right)\right)$$
- **Medyan İmpütasyonu (Aşırı Değerlere Karşı Dayanıklı):**
  $$\hat{x}_{\text{null}} = \text{Median}(X_{\text{dolu}})$$
- **Sınır Değer Kırpması (Clamping):**
  $$\tilde{x} = \min(\max(x, v_{\text{min}}), v_{\text{max}})$$

---

## 🛠️ Dizin Yapısı

```
day-44-pandas-data-quality-cleaner/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # pandas, numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # 3 senaryolu ana yürütme ve test simülasyonu
├── README.md                        # 220+ satır teorik ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── sema.py                      # KolonKurali ve TabloSemasi deklaratif tanımları
│   ├── dogrulayici.py               # SemaDogrulayici (Tip, Sınır, Kategori, Null, Regex)
│   ├── temizleyici.py               # OtomatikVeriTemizleyici (Deduplication, Impute, Clip)
│   └── gorsellestirici.py           # 6 panelli veri kalitesi panosu (Data Quality Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_data_quality.py         # 7 adet birim test (Tümü Başarılı)
└── ciktilar/
    └── veri_kalite_paneli.png       # 6 panelli yüksek çözünürlüklü kalite panosu
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

### 3. Birim Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 📊 3 Üretim Senaryosu ve Teftiş Kararları

| Senaryo | Tablo Özellikleri | Kalite Skoru | Teftiş Kararı & Uygulanan Aksiyon |
|---|---|---|---|
| **1. Golden Table** | 300 Satır, Sıfır Hata | **%100.0** | **GECERLI_MUKEMMEL** (Doğrudan İşlendi) |
| **2. Kirli Tablo** | Negatif Yaş, Eksik Harcama, Mükerrer Satır | **%88.5** | **DUZELTILEBILIR_KIRLI_VERI** $\to$ Otomatik Temizlendi & Kalite %100'e çıkarıldı. |
| **3. Kritik Hata** | `musteri_id` Zorunlu Kolon Eksik | **%0.0** | **KRITIK_RED** (Boru Hattı Güvenle Durduruldu) |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** `src/sema.py` içerisindeki `TabloSemasi` sınıfına JSON Schema formatında dışa aktarım yapan ve harici sistemlerle entegrasyon sağlayan bir **"JSON Schema Exporter"** fonksiyonu eklemek.

**Tamamlanan Çözüm:**
```python
def json_schema_uret(sema: TabloSemasi) -> dict:
    properties = {}
    required = []
    for col_ad, k in sema.kolon_kurallari.items():
        prop = {"type": "number" if k.tip in [int, float] else "string"}
        if k.min_deger is not None:
            prop["minimum"] = k.min_deger
        if k.max_deger is not None:
            prop["maximum"] = k.max_deger
        if k.kategoriler is not None:
            prop["enum"] = k.kategoriler
        properties[col_ad] = prop
        if k.zorunlu:
            required.append(col_ad)
    return {"title": sema.tablo_adi, "type": "object", "properties": properties, "required": required}
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Makine öğrenimi boru hatlarında eksik verileri doldururken (imputation) neden **ortalama (mean)** yerine genellikle **medyan (median)** tercih edilir ve kategorik değişkenlerde mod (mode) yerine özel bir `'BILINMIYOR'` kategorisi açmanın model kararlılığına avantajı nedir?

> **Mentor Cevabı:**
> 1. **Aşırı Değer Hassasiyeti (Outlier Sensitivity):** Ortalama değer, veri setindeki birkaç uç değerden (örneğin geliri 100 milyon TL olan tek bir müşteri) aşırı etkilenir ve tüm eksik değerleri gerçekçi olmayan yüksek sayılarla doldurarak modeli yanıltır. Medyan ise sıralı ortanca değer olduğu için aşırı uç değerlere karşı tamamen dayanıklıdır (robust).
> 2. **Bilgi Kaybını Engelleme:** Kategorik bir değişkendeki eksikliği en sık geçen kategoriyle (mod) doldurmak, verinin "aslında eksik olduğu" bilgisini yok eder. Oysa `'BILINMIYOR'` şeklinde açık bir kategori tanımlandığında, karar ağaçları (XGBoost/LightGBM) eksik olmanın kendisini bir risk faktörü olarak öğrenebilir.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
