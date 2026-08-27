# Day 03: Mahalanobis vs. Öklid Mesafesi ve Çok Değişkenli Dağılım Analizi

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/scipy-1.13+-8CAAE6.svg?style=flat-square&logo=scipy)](https://scipy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; değişkenlerin birbiriyle ilişkili (korelasyonlu) olduğu ve farklı varyanslara sahip bulunduğu çok değişkenli veri dağılımlarında **Öklid mesafesinin neden aldandığını**, **Kovaryans Matrisinin ($\Sigma$) geometrik rolünü** ve **Mahalanobis Mesafesini** sıfırdan vektörize NumPy algoritmalarıyla inceleyerek Ki-Kare ($\chi^2$) temelli endüstriyel anomali tespit motoru sunar.

---

## 📌 Proje Kapsamı ve Matematiksel Mantık

### 1. Öklid'in Kör Noktası
Standart Öklid mesafesi tüm değişkenlerin **bağımsız** olduğunu ve **aynı varyansa** sahip olduğunu varsayar (geometrik olarak daire veya küre şeklinde eş-uzaklık konturları çizer):

$$d_E(x, \mu) = \sqrt{(x - \mu)^T (x - \mu)}$$

Gerçek hayatta (örneğin kumaş üretiminde iplik yoğunluğu ile gramaj, veya finansal piyasalarda iki endeks) değişkenler arasında güçlü bir **korelasyon** vardır. Dağılım bir küre değil, eğik bir **elipsoiddir**.

### 2. Kovaryans Matrisi ($\Sigma$)
Değişkenlerin kendi varyanslarını (köşegen) ve birbirleriyle olan ortak değişimlerini (köşegen dışı) temsil eder:

$$\Sigma = \frac{1}{N - 1} (X - \mu)^T (X - \mu)$$

### 3. Mahalanobis Mesafesi ($D_M$)
Veri kümesinin kovaryans matrisinin tersini ($\Sigma^{-1}$) metriğin kalbine yerleştirerek uzayı beyazlatır (Whitening Transformation). Böylece elipsoidal dağılımı küreselleştirir ve korelasyon yönündeki varyasyonu hesaba katar:

$$D_M(x, \mu) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

```
          Öklid Uzayı                       Mahalanobis (Beyazlatılmış) Uzay
      X2 ▲     ..- -..                           X2 ▲      . - - .
         │   .'       '. (Elips)                    │    .'       '. (Küre)
         │  /   .---.   \                           │   /    (μ)    \
         │ /   /  μ  \   \                          │  |      •      |
         │ \   \     /   /                          │   \           /
         │  \   '---'   /                           │    '.       .'
         │   '.       .'                            │      ' - - '
         └─────────────────► X1                     └─────────────────► X1
      Korelasyonlu Eksenler                         İlişkisiz ve Birim Varyanslı Eksenler
```

---

## 🧮 Ki-Kare ($\chi^2$) ile Anomali Eşiği

$D$ boyutlu çok değişkenli normal dağılımda, Mahalanobis mesafesinin karesi serbestlik derecesi $D$ olan Ki-Kare dağılımına uyar:

$$D_M^2(x, \mu) \sim \chi^2(D)$$

Belirlenen bir anlamlılık düzeyinde ($\alpha = 0.01$, yani %99 güven) kritik eşik değeri $\tau$ şu şekilde hesaplanır:

$$\tau = \sqrt{\chi^2_{1-\alpha}(D)}$$

Eğer bir örneğin $D_M(x, \mu) > \tau$ ise, bu örnek istatistiksel olarak **anomali (üretim kusuru)** kabul edilir.

---

## 📂 Dizin Yapısı

```
day-03-mahalanobis-vs-euclidean/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Teknik dokümantasyon
├── gereksinimler.txt           # Python bağımlılıkları (numpy, scipy, pytest)
├── ana_akis.py                 # Konsol çalıştırma akışı ve deneyler
├── src/
│   ├── __init__.py
│   ├── kovaryans_ve_mesafe.py  # KovaryansAnalizoru & MahalanobisMesafeOlcer
│   └── anomali_tespit_edici.py # Ki-Kare anomali dedektörü
└── testler/
    └── test_mahalanobis.py     # 7 adet birim testi
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
python -m pytest testler/test_mahalanobis.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
