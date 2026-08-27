# Day 02: Vektörel ve Piksel Düzeyinde Mesafe ve Benzerlik Metrikleri

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; bilgisayarlı görüde, semantik görsel aramada, k-En Yakın Komşu (k-NN) algoritmalarında ve derin öğrenme embedding eşleştirmelerinde kullanılan temel matematiksel mesafe ve benzerlik metriklerini (Öklid, Manhattan, Kosinüs, Chebyshev, Minkowski) sıfırdan vektörize NumPy operasyonları ile uygular ve görsel katalog aramasını simüle eder.

---

## 📌 Proje Kapsamı ve Mimari Genel Bakış

Bir görsel arama veya yüz tanıma sistemi, derin öğrenme modelinden çıkan öznitelik vektörlerini (embeddings) karşılaştırarak karar verir. Bu karşılaştırmanın doğruluğu ve hızı seçilen mesafe metriğine ve vektörizasyon kalitesine doğrudan bağlıdır.

```
       Sorgu Vektörü (Q)           Katalog Matrisi (N x D)
              │                               │
              └───────────────┬───────────────┘
                              ▼
                      ┌───────────────┐
                      │  MesafeOlcer  │
                      └───────┬───────┘
                              │
       ┌──────────────┬───────┴───────┬──────────────┐
       ▼              ▼               ▼              ▼
  [Öklid (L2)] [Manhattan (L1)] [Kosinüs Benzerliği] [Chebyshev (L∞)]
  Geometrik    Izgara / Şehir   Açısal Yönelim       Maksimum Boyut
  Uzaklık      Bloku Mesafesi   (Işık Bağımsız)      Farkı
```

---

## 🧮 Matematiksel Metrikler

### 1. Öklid Mesafesi (L2 Normu)
İki nokta arasındaki en kısa doğrusal geometrik mesafedir:

$$d_2(u, v) = \sqrt{\sum_{i=1}^n (u_i - v_i)^2} = \|u - v\|_2$$

### 2. Manhattan Mesafesi (L1 Normu / Şehir Bloku)
Eksenler boyunca mutlak farkların toplamıdır. Aykırı değerlere (outliers) karşı L2 normundan daha dayanıklıdır:

$$d_1(u, v) = \sum_{i=1}^n |u_i - v_i| = \|u - v\|_1$$

### 3. Kosinüs Benzerliği ve Mesafesi
Vektörlerin büyüklüğünden (uzunluğundan) bağımsız olarak aralarındaki açının kosinüsünü ölçer:

$$S_{\cos}(u, v) = \frac{u \cdot v}{\|u\|_2 \cdot \|v\|_2 + \epsilon}, \quad D_{\cos}(u, v) = 1.0 - S_{\cos}(u, v)$$

### 4. Chebyshev Mesafesi (L-Sonsuz Normu)
Koordinatlar arasındaki en büyük mutlak farkı temsil eder:

$$d_\infty(u, v) = \max_i |u_i - v_i| = \|u - v\|_\infty$$

### 5. Genelleştirilmiş Minkowski Mesafesi (Lp Normu)
$$d_p(u, v) = \left(\sum_{i=1}^n |u_i - v_i|^p\right)^{1/p}$$
- $p = 1 \implies$ Manhattan
- $p = 2 \implies$ Öklid
- $p \to \infty \implies$ Chebyshev

---

## 📂 Dizin Yapısı

```
day-02-distance-metrics/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Teknik dokümantasyon
├── gereksinimler.txt           # Python bağımlılıkları
├── ana_akis.py                 # Konsol laboratuvar akışı
├── src/
│   ├── __init__.py
│   ├── mesafe_olcer.py         # MesafeOlcer çekirdek sınıfı
│   └── gorsel_eslestirici.py   # k-NN katalog eşleştirme motoru
└── testler/
    └── test_mesafeler.py       # 8 adet pytest birim testi
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
python -m pytest testler/test_mesafeler.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
