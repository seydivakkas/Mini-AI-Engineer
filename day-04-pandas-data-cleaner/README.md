# Day 04: Üretim Seviyesi Tabüler Veri Temizleme Boru Hattı

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.2+-150458.svg?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; endüstriyel yapay zeka ve bilgisayarlı görü projelerinde sensörlerden veya görüntü metaverilerinden gelen kirli, eksik ve aykırı tabüler verileri **veri sızıntısına (Data Leakage) izin vermeyen `fit` - `transform` mimarisiyle** temizleyen, IQR tabanlı **Winsorization** uygulayan ve bellek tüketimini güvenli **tip daraltma (Downcasting)** ile optimize eden üretim seviyesinde bir veri temizleme boru hattıdır.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Yapay zeka modelleri sadece ham piksellerle değil; kameranın ISO değeri, pozlama süresi, sensör sıcaklığı, üretim bandı hızı gibi **tabüler metaverilerle** de beslenir.
Endüstriyel veri boru hatlarında yapılan en büyük ölümcül hata:
> *"Tüm veriyi (eğitim ve test dahil) birleştirip `df.fillna(df.mean())` yapmak!"*

Bu işlem **Veri Sızıntısına (Data Leakage)** yol açar. Test kümesindeki geleceğe ait bilgi eğitim kümesine sızar. Model yerel testlerde %99 doğruluk verirken canlı üretim ortamında çöker!

---

#

---

### 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Tabüler veri kümelerindeki eksik kayıtları, aykırı değerleri ve tutarsız veri tiplerini deterministik kurallarla temizlemek için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Bozuk veya kirli verilerin makine öğrenimi boru hatlarına girerek modelleri zehirlemesini ve çökmelere yol açmasını engeller.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Büyük ölçekli (100GB+) veri kümelerinde tek makine bellek kısıtlarına takılır.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- Polars, PySpark, Dask veya DuckDB.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Eksik Veri Doldurma** | *Missing Value Imputation* | Tablolardaki `NaN` ve `None` boşluklarının ortalama, medyan, mod veya model tabanlı tahmin yöntemleriyle doldurulması. |
| **Tip Dönüşümü & Zorlama** | *Type Coercion & Casting* | Veri tiplerinin bellek verimliliği ve hesaplama doğruluğu için güvenli şekilde dönüştürülmesi (`astype`, `to_numeric`). |
| **Zincirleme Atama Uyarısı** | *SettingWithCopyWarning* | Pandas'ta bir DataFrame görünümü (view) üzerinde doğrudan atama yapıldığında orijinal tablonun güncellenmeme riskini belirten uyarı. |
| **Aykırı Değer Maskeleme** | *Outlier Masking & Clipping* | Aşırı uç değerlerin belirli persentil eşiklerine sıkıştırılması (Winsorizing) veya filtre ile temizlenmesi. |
| **Vektörize Metin Temizleme** | *Vectorized String Operations* | Pandas `.str` erişicisi ile Python döngüleri yerine C seviyesinde optimize edilmiş regex ve metin operasyonları yürütülmesi. |

---

## 2. Matematiksel ve Algoritmik Mantık

#### A. Veri Sızıntısına Karşı `fit()` ve `transform()` Ayrımı
- `fit(egitim_verisi)`: Yalnızca eğitim kümesindeki istatistikleri (medyan, mod, IQR sınırları) öğrenir ve dahili belleğinde saklar.
- `transform(yeni_veri)`: Daha önce öğrenilen bu sabit istatistikleri yeni gelen test veya canlı üretim verisine uygular; asla yeni verinin ortalamasını hesaplamaz!

#### B. IQR Tabanlı Winsorization (Aykırı Değerleri Baskılama)
Aykırı değerleri doğrudan silmek (`drop`) değerli verileri yok eder. Bunun yerine veriyi istatistiksel sınırlarla kırparız:
- Birinci Çeyrek ($Q_1$ / %25) ve Üçüncü Çeyrek ($Q_3$ / %75)
- Çeyrekler Açıklığı: $IQR = Q_3 - Q_1$
- Alt Sınır: $Alt = Q_1 - 1.5 \times IQR$
- Üst Sınır: $Ust = Q_3 + 1.5 \times IQR$
Bu sınırların dışındaki tüm değerler sınırlara çekilir (clipping / winsorization).

#### C. Güvenli Tip Daraltma (Downcasting) ile Bellek Optimizasyonu
Pandas varsayılan olarak her tamsayıyı `int64` (8 bayt) ve her ondalıklı sayıyı `float64` (8 bayt) olarak açar. 
- Eğer bir sütundaki değerler $0$ ile $100$ arasındaysa bu sütun `uint8` (1 bayt) olarak saklanabilir.
- Değer aralıkları taranarak veri kaybı olmadan yapılan bu dönüştürme, RAM tüketimini **%50-%70** oranında düşürür.

---

### 3. Dikkat Edilmesi Gereken Kritik Tuzaklar

1. **Körlemesine `dropna()` Yapmak:** Eksik verisi olan satırları doğrudan silmek, endüstriyel sensör kesintilerinde verinin yarısını çöpe atabilir ve dağılımı yanlı (biased) hale getirebilir.
2. **Kategorik Verilerde "Bilinmeyen" Sınıfı:** Canlı ortamda daha önce hiç görülmemiş bir kategori geldiğinde sistem çökmek yerine bunu `"BILINMEYEN"` etiketiyle karşılamalıdır.

---

## 📌 Mimari Tasarım ve Akış Şeması

```
        Ham Kirli Veri (DataFrame)
                    │
                    ▼
       ┌────────────────────────┐
       │ TabulerVeriTemizleyici │
       └───────────┬────────────┘
                   │
    ┌──────────────┼──────────────┬──────────────────┐
    ▼              ▼              ▼                  ▼
[Kayıt Tekrarı] [Eksik Değer]  [Winsorization]   [Tip Daraltma]
- Çift kayıtlar  - Sayısal:      - IQR Alt/Üst      - int64 -> int16
  temizlenir       Medyan         Sınır Kırpma      - float64 -> float32
                 - Kategorik:                       - %50+ RAM Tasarrufu
                   Mod
```

---

## 💻 Konsol Çalıştırma Çıktısı

```text
======================================================================
>>> AŞAMA 1: Kirli Sentetik Endüstriyel Veri Seti Üretimi
======================================================================
[+] Üretilen Satır Sayısı           : 1000
[+] Sütunlar                        : ['kamera_id', 'pozlama_suresi_ms', 'sensor_sicakligi_c', 'isik_parlakligi_lumen', 'kalite_etiketi']
[+] Başlangıç Bellek Tüketimi       : 39.19 KB
[+] Eksik Değer Özeti               : 
    - pozlama_suresi_ms: 50 eksik
    - sensor_sicakligi_c: 40 eksik
    - isik_parlakligi_lumen: 60 eksik
    - kalite_etiketi: 30 eksik
[+] Mükerrer Satır Sayısı           : 30

======================================================================
>>> AŞAMA 2: Eğitim ve Test Kümelerine Bölme (Data Leakage Önleme)
======================================================================
[+] Eğitim Kümesi Boyutu            : (800, 5)
[+] Test Kümesi Boyutu              : (200, 5)

======================================================================
>>> AŞAMA 3: Temizleme Boru Hattının Eğitilmesi (fit)
======================================================================
Öğrenilen Sayısal Medyanlar:
  * pozlama_suresi_ms        : 31.95
  * sensor_sicakligi_c       : 45.02
  * isik_parlakligi_lumen    : 799.30
Öğrenilen Kategorik Modlar:
  * kalite_etiketi           : KUSURSUZ

======================================================================
>>> AŞAMA 4: Eğitim Kümesinin Dönüştürülmesi (transform)
======================================================================
[+] Temizlik Sonrası Kalan Eksik Değer: 0
[+] Temizlik Sonrası Mükerrer Satır   : 0
[+] Önceki Bellek: 31.38 KB -> Yeni Bellek: 15.60 KB
[V] Bellek Tasarrufu: %50.3

======================================================================
>>> AŞAMA 5: Test Kümesinin Dönüştürülmesi (Sızıntısız Test)
======================================================================
[+] Test Kümesi Başarıyla Temizlendi! Kalan Eksik Değer: 0
```

---

## 🎯 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

🎯 **Görevin: Eksik Değer Bayrağı (Missing Indicator Feature) Eklemek**

Bazen bir değerin eksik olması rastgele değildir; bir arızanın veya kritik bir durumun habercisidir. Bu bilgiyi kaybetmemek için doldurulan her eksik sütun için `sutun_adi_eksik_mi` adında boolean ($0$ veya $1$) bir gösterge sütunu üretilir.

### Görev Tanımı:
[`src/veri_temizleyici.py`](./src/veri_temizleyici.py) sınıfının `transform()` metoduna `eksiklik_gostergesi_ekle=True` parametresi ekle. Sütun doldurulmadan önce nerede `NaN` olduğunu kaydeden yeni bir ikili (binary) sütun oluştur.

---

## 🧠 Gün Sonu Kontrol Noktası & Mentorun Teknik Sorusu

> **Teknik Soru:**  
> Bir veri temizleme boru hattında `fit()` metodunu neden **asla test kümesi üzerinde çalıştırmamalıyız**? Test kümesindeki eksik değerleri eğitim kümesinin medyanı ile doldurmak neden doğru olan tek yöntemdir?

---

## 📂 Dizin Yapısı

```
day-04-pandas-data-cleaner/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Kapsamlı ders ve teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar (pandas, numpy, pytest)
├── ana_akis.py                 # Konsol laboratuvar akışı
├── src/
│   ├── __init__.py
│   ├── sentetik_veri_ureticisi.py # Kirli veri jeneratörü
│   └── veri_temizleyici.py     # TabulerVeriTemizleyici sınıfı
└── testler/
    └── test_temizleyici.py     # 7 adet birim testi (7 passed)
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleme
```bash
pip install -r gereksinimler.txt
```

### 2. Ana Laboratuvar Akışını Çalıştırma
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
