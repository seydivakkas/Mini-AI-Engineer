# Day 05: Otomatik Veri Seti Profilleme ve Özet Raporlama Motoru

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.2+-150458.svg?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/scipy-1.13+-8CAAE6.svg?style=flat-square&logo=scipy)](https://scipy.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; ağır ve harici bağımlılıkları çok olan profilleme kütüphanelerine (ydata-profiling, sweetviz vb.) ihtiyaç duymadan, doğrudan **Pandas, NumPy ve SciPy** ile tabüler yapay zeka metaverilerini saniyeler içinde analiz eden, **sütun kardinalitesi**, **çarpıklık (skewness)**, **basıklık (kurtosis)**, **kayıp veri oranları** ve **otomatik anlamsal tip sınıflandırması** üreten hafif (lightweight) bir MLOps profilleme motoru sunar.

---

## 📌 Proje Kapsamı ve Mimari Genel Bakış

Bir veri bilimi veya MLOps hattında yeni bir veri seti geldiğinde ilk adım verinin karakteristiğini çıkarmaktır:
- Veri setinde hangi sütunlar benzersiz kimlik (ID)?
- Hangi sayısal sütunlar normal dağılıyor, hangileri aşırı sağa/sola çarpık?
- Veri tabanında gereksiz yer kaplayan sıfır varyanslı (sabit) kolonlar var mı?
- Eksik veri oranları kritik eşiği (%20) aşıyor mu?

```
                      Ham Tablo (pd.DataFrame)
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  MiniVeriProfilleyici │
                     └───────────┬───────────┘
                                 │
       ┌────────────────┬────────┴────────┬────────────────┐
       ▼                ▼                 ▼                ▼
[Kardinalite]    [Dağılım Şekli]   [Kayıp Veri]    [Anlamsal Tip & Uyarı]
Benzersiz Değer  Çarpıklık (Skew)  Eksiklik Oranı  - Sabit Sütun (Var=0)
Oranı (nunique)  Basıklık (Kurt)   Hücre Sayısı    - Aşırı Çarpık Dağılım
                                                   - Aday ID Anahtarı
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ ProfilRaporOlusturucu │
                     └───────────┬───────────┘
                                 │
                     ┌───────────┴───────────┐
                     ▼                       ▼
             [Konsol Tablosu]        [Markdown / JSON]
```

---

## 🧮 İstatistiksel ve Matematiksel Mantık

### 1. Sütun Kardinalitesi (Tekillik Oranı)
Bir sütundaki benzersiz değer sayısının ($U$) toplam geçerli satır sayısına ($N$) oranıdır:

$$\text{Kardinalite Oranı} = \frac{U}{N}$$

- Oran $> 0.95$ ise $\to$ **Aday Benzersiz Anahtar (ID)** veya serbest metindir (model eğitiminde doğrudan kullanılmamalı, elenmeli veya hash'lenmelidir).
- Oran $< 0.05$ ise $\to$ **Kategorik veya Ayrık Değişken** adayıdır.

### 2. Çarpıklık (Skewness / Üçüncü Standartlaştırılmış Moment)
Veri dağılımının simetriden ne kadar saptığını gösterir:

$$\tilde{\mu}_3 = \frac{\frac{1}{N} \sum_{i=1}^N (x_i - \mu)^3}{\sigma^3}$$

- **$\tilde{\mu}_3 = 0$:** Mükemmel simetrik (Normal dağılım).
- **$\tilde{\mu}_3 > +1.5$:** Sağa (pozitif) çarpık / kalın sağ kuyruk (ör. dosya boyutları, gelirler). Logaritmik dönüşüm $\log(1 + x)$ önerilir.
- **$\tilde{\mu}_3 < -1.5$:** Sola (negatif) çarpık.

### 3. Basıklık (Kurtosis / Dördüncü Standartlaştırılmış Moment)
Dağılımın kuyruk kalınlığını ve sivrilik derecesini ölçer (Fisher tanımı ile normal dağılım = 0):

$$\tilde{\mu}_4 = \frac{\frac{1}{N} \sum_{i=1}^N (x_i - \mu)^4}{\sigma^4} - 3$$

- **$> 0$ (Leptokurtic):** Kalın kuyruklu, aykırı değer üretme olasılığı çok yüksek.
- **$< 0$ (Platykurtic):** İnce kuyruklu, tekdüze dağılıma yakın.

---

## 📂 Dizin Yapısı

```
day-05-mini-data-profiler/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar (pandas, numpy, scipy, pytest)
├── ana_akis.py                 # Konsol ve Markdown profilleme akışı
├── src/
│   ├── __init__.py
│   ├── veri_profilleyici.py     # MiniVeriProfilleyici ve veri yapıları
│   └── rapor_olusturucu.py     # Konsol ve Markdown raporlayıcı
└── testler/
    └── test_profilleyici.py    # 6 adet pytest birim testi
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
python -m pytest testler/test_profilleyici.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
