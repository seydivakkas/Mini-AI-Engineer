# Day 03: Mahalanobis vs. Öklid Mesafesi & Çok Değişkenli Anomali Tespiti

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/scipy-1.13+-blue.svg?style=flat-square&logo=scipy)](https://scipy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; çok değişkenli veri dağılımlarında değişkenler arasındaki kovaryans, korelasyon ve farklı ölçekleri hesaba katan **Mahalanobis Mesafesi** ile klasik **Öklid Mesafesi** arasındaki farkları inceler, tekil matris riskine karşı **Tikhonov (Ridge) Düzenlileştirmesi** uygular ve Ki-Kare ($\chi^2$) dağılımı ile endüstriyel kalite kontrolde çok boyutlu aykırı değer/anomali tespiti gerçekleştirir.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Endüstrideki Yeri ve Çözdüğü Temel Problem
Endüstriyel bir üretim bandında bir parçanın "kusurlu/anormal" olup olmadığını tespit etmek istediğimizi düşünelim. İki sensörümüz olsun:
- $X_1$: Parçanın uzunluğu (cm)
- $X_2$: Parçanın ağırlığı (kg)

Doğal olarak uzunluk arttıkça ağırlık da artar; yani aralarında **güçlü bir pozitif korelasyon** vardır.
Şimdi bir parça geldi: Boyu çok uzun ($300\text{ cm}$), ama ağırlığı tüy gibi hafif ($10\text{ kg}$).
Klasik **Öklid mesafesi**, veri noktalarının birbirine olan dairesel/küresel mesafesine bakar; değişkenler arasındaki bu güçlü korelasyonu ve ölçek farkını kesinlikle **göremez**. Noktanın ortalamaya olan Öklid mesafesi normal sınırlar içinde kalabilir ve sistem bu kusurlu parçayı kaçırır!

İşte **Mahalanobis Mesafesi**, verinin kovaryans elipsini ve eksenlerin birbirine olan eğimini temel alarak mesafeyi ölçer.

---

#

---

### 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Veri dağılımının kovaryans matrisini ($S^{-1}$) hesaba katarak değişkenler arasındaki korelasyonu ve ölçek farklılıklarını nötralize eden mesafe hesaplamak için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Öklid mesafesinin korelasyonlu ve eliptik saçılan verilerde ürettiği yanıltıcı mesafe ölçüm hatalarını tamamen çözer.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Kovaryans matrisinin tekil (singular/non-invertible) olduğu durumlarda tersi alınamaz; düzenlileştirme (shrinkage) gerektirir.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- PCA ile beyazlatılmış (Whitened) Öklid Mesafesi veya Wasserstein Mesafesi.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Mahalanobis Mesafesi** | *Mahalanobis Distance* | Değişkenler arasındaki korelasyonu ve her eksendeki varyans farkını kovaryans matrisi ($\Sigma$) ile hesaba katan istatistiksel mesafe metriği ($D_M = \sqrt{(x-\mu)^T \Sigma^{-1} (x-\mu)}$). |
| **Kovaryans Matrisi** | *Covariance Matrix* | Çok değişkenli bir veri setinde tüm öznitelik çiftlerinin birlikte nasıl değiştiklerini gösteren simetrik pozitif yarı-tanımlı matris. |
| **Ters Kovaryans (Hassasiyet)** | *Inverse Covariance / Precision Matrix* | Kovaryans matrisinin tersi ($\Sigma^{-1}$) olup, değişkenler arasındaki koşullu bağımsızlık yapısını ve uzaydaki eksen ölçeklemesini belirler. |
| **Korelasyon Duyarlılığı** | *Correlation Invariance* | Öznitelikler arasındaki doğrusal bağımlılıkların mesafeyi yapay olarak şişirmesini veya saptırmasını önleme yeteneği. |
| **Çok Değişkenli Aykırı Değer** | *Multivariate Outlier* | Tekil eksenlerde normal görünen ancak özniteliklerin ortak dağılımında (eliptik sınırların dışında) kalan sinsi anomali noktaları. |

---

## 2. Matematiksel ve Algoritmik Mantık

#### A. Öklid vs. Mahalanobis Karşılaştırması
- **Öklid Mesafesi:**
  $$d_{\text{oklid}}(x, \mu) = \sqrt{(x - \mu)^T (x - \mu)}$$
  Eşit mesafe konturları mükemmel birer **küredir (çemberdir)**. Eksenlerin bağımsız ve aynı varyansa sahip olduğunu varsayar.

- **Mahalanobis Mesafesi:**
  $$D_M(x, \mu) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$
  *(Burada $\Sigma$, verinin $D \times D$ boyutlu Kovaryans Matrisidir).*
  - Eşit mesafe konturları verinin yönüne doğru uzayan **elipsoidlerdir**.
  - Veriyi önce kovaryans eksenlerine göre döndürür, ardından her ekseni kendi standart sapmasına bölerek ölçekler.

#### B. Tekil Matris Problemi ve Tikhonov Düzenlileştirmesi
Eğer veride iki değişken birbiriyle tam doğrusal bağımlıysa (ör. $x_2 = 2 x_1$) veya örneklem sayısı boyut sayısından azsa ($N < D$), kovaryans matrisinin determinantı sıfır olur ($\det(\Sigma) = 0$). Yani matrisin tersi ($\Sigma^{-1}$) **alınamaz**!

Bunu çözmek için matrisin köşegenine küçük bir düzenlileştirme katsayısı eklenir:
$$\Sigma_{\text{duzenli}} = \Sigma + \lambda I$$

#### C. Ki-Kare ($\chi^2$) Dağılımı ile İstatistiksel Eşik Belirleme
Eğer veri çok değişkenli normal dağılıma uyuyorsa, Mahalanobis mesafesinin karesi ($D_M^2$), serbestlik derecesi değişken sayısı ($D$) olan bir **Ki-Kare ($\chi^2_D$) dağılımına** uyar:
$$D_M^2 \sim \chi^2_D$$
Belirlenen bir anlamlılık düzeyinde ($\alpha = 0.01$ yani %99 güvenle), eşik değeri kütüphaneden doğrudan teorik olarak çekilir:
$$\text{Eşik} = \sqrt{\chi^2_{D, 1 - \alpha}}$$

---

### 3. Dikkat Edilmesi Gereken Kritik Tuzaklar

1. **Ölçek Yanılgısı:** Öklid mesafesi birimi milimetre olan bir sütun ile kilogram olan bir sütunu toplar. Değerleri büyük olan sütun Öklid mesafesini domine eder. Mahalanobis ise varyansa bölerek otomatik standardize eder.
2. **Korelasyona Körlük:** Öklid mesafesi değişkenlerin birbirini etkilemediğini varsayar. Gerçek dünya verilerinde neredeyse tüm sensörler birbiriyle ilişkilidir.

---

## 📌 Mimari Tasarım ve Akış Şeması

```
     Çok Değişkenli Veri Matrisi (N x D)
                      │
                      ▼
     ┌─────────────────────────────────┐
     │  KovaryansVeMahalanobisHesaplayici │
     └────────────────┬────────────────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
[Kovaryans Matrisi Σ]      [Tikhonov Düzenlileştirme]
- Örneklem Kovaryansı      - Σ + λI (Ters Alınabilirliği
- Korelasyon Matrisi R       Garanti Eder)
        │                           │
        └─────────────┬─────────────┘
                      ▼
          ┌───────────────────────┐
          │ AnomaliTespitEdici    │
          │ (Ki-Kare Eşik Testi)  │
          └───────────────────────┘
```

---

## 💻 Konsol Çalıştırma Çıktısı

```text
======================================================================
>>> AŞAMA 1: Korelasyonlu Normal Veri ve Aykırı Noktalar Üretimi
======================================================================
[+] Normal Örneklem Sayısı : 600
[+] Boyut (Öznitelik) Sayısı: 2

======================================================================
>>> AŞAMA 2: Kovaryans ve Korelasyon Matrislerinin İncelenmesi
======================================================================
Hesaplanan Kovaryans Matrisi:
 [[15.82 23.11]
  [23.11 35.40]]
Hesaplanan Korelasyon Matrisi (Pearson):
 [[1.   0.98]
  [0.98 1.  ]]
[!] İki sensör arasında %98 oranında çok güçlü pozitif korelasyon var!

======================================================================
>>> AŞAMA 3: Öklid vs. Mahalanobis Mesafesi Kıyaslama Deneyi
======================================================================
--------------------------------------------------------------------------------
Nokta Türü             | Koordinat (X1, X2)     | Öklid Mesafesi | Mahalanobis    
--------------------------------------------------------------------------------
Merkez Nokta           | [100.0, 50.0]          | 0.00           | 0.00           
Trend İçi Uzak Nokta   | [112.0, 68.0]          | 21.63          | 3.06           
Trend Dışı Anomali     | [112.0, 32.0]          | 21.63          | 24.81          
--------------------------------------------------------------------------------
[!] GÖZLEM:
  - 'Trend İçi' ve 'Trend Dışı' noktaların Merkez'e olan Öklid mesafeleri (21.63)
    BİREBİR AYNIDIR! Öklid korelasyonu ve yönü göremez!
  - Mahalanobis ise Trend Dışı noktaya 24.81 mesafe vererek
    onu anında KRİTİK BİR ANOMALİ olarak damgalamıştır!
```

---

## 🎯 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

🎯 **Görevin: İki Bağımsız Dağılım Arasında Mahalanobis Mesafesi**

Bugünkü kodumuz tek bir noktanın bir dağılımın merkezine olan mesafesini ($D_M(x, \mu)$) hesaplıyor. İki farklı sınıfın (ör. Sağlam Kumaşlar $\mu_1, \Sigma_1$ vs. Kusurlu Kumaşlar $\mu_2, \Sigma_2$) dağılımları arasındaki mesafeyi ölçmek için **Birleşik (Pooled) Kovaryans Matrisi** kullanılır:

$$\Sigma_{\text{birlesik}} = \frac{(N_1 - 1)\Sigma_1 + (N_2 - 1)\Sigma_2}{N_1 + N_2 - 2}$$
$$D_M(\mu_1, \mu_2) = \sqrt{(\mu_1 - \mu_2)^T \Sigma_{\text{birlesik}}^{-1} (\mu_1 - \mu_2)}$$

### Görev Tanımı:
[`src/kovaryans_ve_mesafe.py`](./src/kovaryans_ve_mesafe.py) içerisine `iki_kume_arasi_mahalanobis()` adında bir metod ekle ve iki farklı dağılım kümesinin ayrışma derecesini test et.

---

## 🧠 Gün Sonu Kontrol Noktası & Mentorun Teknik Sorusu

> **Teknik Soru:**  
> Kovaryans matrisi $\Sigma$ köşegen bir matris (diagonal matrix) olduğunda (yani tüm değişkenlerin kovaryansı $0$, sadece kendi varyansları $\sigma_i^2$ mevcut olduğunda), **Mahalanobis Mesafesi** hangi iyi bilinen normalize edilmiş Öklid formülüne dönüşür?

---

## 📂 Dizin Yapısı

```
day-03-mahalanobis-vs-euclidean/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Kapsamlı ders ve teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar (numpy, scipy, pytest)
├── ana_akis.py                 # Konsol çalıştırma akışı
├── src/
│   ├── __init__.py
│   ├── kovaryans_ve_mesafe.py  # Kovaryans, Mahalanobis ve Tikhonov
│   └── anomali_tespit_edici.py # Ki-Kare anomali tespit motoru
└── testler/
    └── test_mahalanobis.py     # 7 adet birim testi (7 passed)
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
