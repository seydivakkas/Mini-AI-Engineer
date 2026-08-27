# Day 01: NumPy Görüntü Analizörü ve Piksel İstatistikleri

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje, harici görüntü işleme kütüphanelerine (OpenCV, Pillow vb.) bağımlı kalmadan, doğrudan **NumPy** çok boyutlu dizi (ndarray) mimarisi üzerinden dijital görüntülerin piksel düzeyinde incelenmesini, renk kanalı ayrıştırmasını, ağırlıklı gri tonlama dönüşümünü, istatistiksel profil çıkarımını ve derin öğrenme modellerine hazır normalizasyon boru hatlarını sıfırdan inşa eder.

---

## 📌 Proje Kapsamı ve Mimari Genel Bakış

Dijital bir görüntü temelde $H \times W \times C$ boyutlarında bir tensördür:
- **$H$ (Yükseklik - Height):** Matris satır sayısı (Y ekseni).
- **$W$ (Genişlik - Width):** Matris sütun sayısı (X ekseni).
- **$C$ (Kanal Sayısı - Channels):** Renk derinliği (Gri: 1, RGB: 3, RGBA: 4).

```
   Girdi Görsel Matrisi (H x W x C, uint8)
                    │
                    ▼
       ┌────────────────────────┐
       │  NumPyGoruntuAnalizoru │
       └───────────┬────────────┘
                   │
    ┌──────────────┼──────────────┬──────────────────┐
    ▼              ▼              ▼                  ▼
[Kanal Ayrıştırma] [Gri Dönüşüm] [İstatistikler]   [Normalizasyon]
- Kırmızı Matrisi  - BT.601       - Min, Max, Ort.   - Min-Max [0, 1]
- Yeşil Matrisi    - Ağırlıklı    - Medyan, Varyans  - Min-Max [-1, 1]
- Mavi Matrisi       Lüminans     - Çeyreklikler     - Z-Skoru N(0, 1)
```

---

## 🧮 Matematiksel Temeller

### 1. Ağırlıklı Lüminans Gri Ton Dönüşümü (ITU-R BT.601)
İnsan gözü yeşil renge en yüksek, maviye ise en düşük fotoreseptör duyarlılığına sahiptir. Bu sebeple basit aritmetik ortalama yerine algısal ağırlıklandırma kullanılır:

$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

### 2. Min-Max Doğrusal Normalizasyonu
Piksel değerlerini belirlenen $[a, b]$ aralığına ölçekler ($\epsilon = 10^{-8}$ sıfıra bölmeyi engeller):

$$X_{\text{norm}} = a + \frac{X - X_{\min}}{(X_{\max} - X_{\min}) + \epsilon} \cdot (b - a)$$

### 3. Z-Skoru Standartlaştırması (Kanal Bazlı)
Dağılımı $\mu = 0$ ve $\sigma = 1$ olan standart normal dağılıma dönüştürür:

$$Z_c = \frac{X_c - \mu_c}{\sigma_c + \epsilon}$$

---

## 📂 Dizin Yapısı

```
day-01-numpy-image-analyzer/
├── LICENSE                     # Özel Tüm Hakları Saklıdır lisans dosyası
├── README.md                   # Proje dokümantasyonu
├── gereksinimler.txt           # Python bağımlılıkları
├── ana_akis.py                 # Konsol çalıştırma betiği
├── src/
│   ├── __init__.py             # Paket tanımlayıcısı
│   ├── goruntu_analizoru.py    # NumPyGoruntuAnalizoru çekirdek sınıfı
│   └── yardimcilar.py          # Sentetik görsel ve bellek araçları
└── testler/
    └── test_analizor.py        # Kapsamlı pytest birim testleri
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
python -m pytest testler/test_analizor.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
