# Day 06: Kapsamlı Keşifçi Veri Analizi Laboratuvarı (EDA Lab)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.2+-150458.svg?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.9+-11557c.svg?style=flat-square&logo=matplotlib)](https://matplotlib.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; bilgisayarlı görü ve yapay zeka öznitelik tablolarında **doğrusal (Pearson) ve monotonik (Spearman) korelasyon analizlerini**, makine öğrenimi modellerini zehirleyen **Çoklu Doğrusallık (Multicollinearity)** problemini **Varyans Şişme Faktörü (VIF)** ile tespit etmeyi ve **otomatik grafik görselleştirmelerini (Isı Haritası, Histogram, Saçılım)** sunucu dostu (headless) olarak üreten bir keşifçi veri analizi laboratuvarıdır.

---

## 📌 Proje Kapsamı ve Mimari Genel Bakış

Bir model eğitilmeden önce öznitelikler arasındaki gizli ilişkilerin açığa çıkarılması gerekir:
- İki veya daha fazla öznitelik birbiriyle neredeyse %100 aynı bilgiyi taşıyorsa (ör. iplik sıklığı ile toplam düğüm sayısı), model ağırlıkları sayısal kararsızlığa uğrar.
- Hangi öznitelikler hedef değişkeni (kusurlu alan) en güçlü şekilde açıklıyor?
- Doğrusal olmayan ama monotonik olan ilişkiler nasıl yakalanır?

```
                     Öznitelik Veri Çerçevesi (pd.DataFrame)
                                        │
                                        ▼
                           ┌─────────────────────────┐
                           │   KesifciVeriAnalizoru  │
                           └────────────┬────────────┘
                                        │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
[Pearson / Spearman]     [VIF Analizi]          [Hedef Değişken]
Doğrusal vs. Monotonik   Çoklu Doğrusallık      Öznitelik Korelasyonları
Korelasyon Matrisleri    (1 / (1 - R^2))        Kategori Dağılımları
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                   ┌─────────────────────────┐
                   │    EdaGrafikUreteci     │
                   │   (Headless Matplotlib) │
                   └────────────┬────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
[korelasyon_isi_        [ozellik_               [dokuma_hizi_vs_kusur_
 haritasi.png]           dagilimlari.png]        sacilim.png]
```

---

## 🧮 İstatistiksel ve Matematiksel Mantık

### 1. Pearson vs. Spearman Korelasyonu
- **Pearson ($r$):** Değişkenler arasındaki **doğrusal ilişkiyi** ölçer:
  $$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
- **Spearman ($\rho$):** Değerlerin kendisi yerine sıralamalarını (ranks) kullanarak **monotonik (artarken artan ama doğrusal olmayan) ilişkileri** ölçer.
- **Kritik Kural:** Eğer Pearson düşük ama Spearman yüksekse, değişkenler arasında **doğrusal olmayan ama güçlü bir kural/ilişki** (ör. üstel veya logaritmik) vardır!

### 2. Çoklu Doğrusallık ve Varyans Şişme Faktörü (VIF)
Bir bağımsız değişkenin ($X_i$), diğer tüm bağımsız değişkenler tarafından ne kadar açıklandığını En Küçük Kareler (OLS) $R_i^2$ skoru ile ölçer:

$$VIF_i = \frac{1}{1 - R_i^2}$$

- **$VIF < 5$:** Güvenli ve sağlıklı öznitelik.
- **$5 \le VIF \le 10$:** Orta riskli doğrusallık.
- **$VIF > 10$:** **Kritik Çoklu Doğrusallık.** Modelin regresyon katsayıları şişer, test kümesinde genelleme yeteneği çöker. Bu özniteliklerden biri modelden çıkarılmalıdır.

---

## 📂 Dizin Yapısı

```
day-06-eda-lab/
├── LICENSE                     # Özel Tüm Hakları Saklıdır Lisansı
├── README.md                   # Teknik dokümantasyon
├── gereksinimler.txt           # Bağımlılıklar (pandas, numpy, scipy, matplotlib, pytest)
├── ana_akis.py                 # Konsol ve grafik çalıştırma betiği
├── ciktilar/                   # Üretilen yüksek çözünürlüklü grafikler
│   ├── korelasyon_isi_haritasi.png
│   ├── ozellik_dagilimlari.png
│   └── dokuma_hizi_vs_kusur_sacilim.png
├── src/
│   ├── __init__.py
│   ├── kesifci_analizor.py     # KesifciVeriAnalizoru ve VIF motoru
│   └── grafik_ureteci.py       # Headless görselleştirme motoru
└── testler/
    └── test_eda.py             # 5 adet pytest birim testi
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
python -m pytest testler/test_eda.py -v
```

---

## 🔒 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır.
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). Ayrıntılar için [LICENSE](./LICENSE) dosyasını inceleyiniz.
