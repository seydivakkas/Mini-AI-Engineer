# Day 38: Halı Doku ve Desenleri İçin Çoklu Özellikli Görsel Arama (Carpet Visual Retrieval)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6.svg?style=flat-square&logo=scipy)](https://scipy.org/)
[![Pillow](https://img.shields.io/badge/Pillow-9.5+-005571.svg?style=flat-square)](https://python-pillow.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; halı ve tekstil kataloglarında, e-ticaret platformlarında ve tasarım stüdyolarında kullanıcıların yüklediği bir halı görseline en yakın ürünleri getiren **Çoklu Özellikli Görsel Arama Motorudur (Multi-Feature Visual Retrieval Engine)**. Halı görselini yalnızca renge veya yalnızca şekle göre aramak yetersiz olduğundan; **3B HSV Renk Histogramı + Renk Momentleri** ile **GLCM Haralick Doku İstatistikleri + LBP Mikro-Doku Tanımlayıcılarını** ağırlıklı füzyonla birleştirir.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Neden Halı/Tekstil Aramalarında Çoklu Özellik Füzyonu Zorunludur?
* **Yalnızca Renk Tabanlı Arama:** Aynı antrasit renkte olan ancak biri *kabarık uzun tüylü Shaggy* halı, diğeri *düz dokuma ince Jüt hasır* halı olan iki ürünü ayırt edemez.
* **Yalnızca Doku/Kenar Tabanlı Arama:** Aynı geometrik desene sahip ancak biri *pastel bej-hardal*, diğeri *canlı kırmızı-mavi* olan iki ürünü ayırt edemez.
* **Çoklu Özellik Füzyonu (Multi-Feature Fusion):** Renk bilgisi ile dokuma türü / hav yüksekliği / mikro-örgü yapısını tek bir ortak benzerlik uzayında birleştirir.

```
                    ┌──────────────────────────────────────────────────────────┐
                    │               SORGU HALISI GÖRSELİ (RGB)                 │
                    └────────────────────────────┬─────────────────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
        ┌───────────────────────────────┐                 ┌───────────────────────────────┐
        │   1. RENK ÖZELLİK ÇIKARICI    │                 │   2. DOKU ÖZELLİK ÇIKARICI    │
        │ - 3B HSV Histogramı (128-d)   │                 │ - GLCM Haralick (4 Yön: 20-d) │
        │ - Renk Momentleri (Ort,Std,Sk)│                 │ - LBP Mikro-Doku (16-d)       │
        │ - L2 Normalizasyon (v_renk)   │                 │ - L2 Normalizasyon (v_doku)   │
        └───────────────┬───────────────┘                 └───────────────┬───────────────┘
                        │                                                 │
                        └────────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │  3. AĞIRLIKLI KOSİNÜS BENZERLİK FÜZYONU (Multi-Feature Weighted Cosine)      │
        │  S_hibrit = w_renk * S(v_q_renk, v_d_renk) + w_doku * S(v_q_doku, v_d_doku) │
        └────────────────────────────────────────┬─────────────────────────────────────┘
                                                 │
                                                 ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │  4. TOP-K KATALOG SIRALAMASI VE GÖRSEL BENZERLİK RAPORU                      │
        │  Rank 1: Hereke Klasik Bordo (%98.4)                                         │
        │  Rank 2: Anadolu Eskitme Terracotta (%74.2)                                  │
        └──────────────────────────────────────────────────────────────────────────────┘
```

---

### 2. GLCM (Gray-Level Co-occurrence Matrix) ve Haralick İstatistikleri

GLCM, bir görüntüde aralarında $d$ mesafe ve $\theta$ açı bulunan piksel çiftlerinin gri seviye eş-oluşum frekansını sayar. Çıkarılan 5 temel istatistik:

1. **Kontrast (Contrast):** Lokal yoğunluk değişimlerini ve pürüzlülüğü ölçer:
   $$\text{Kontrast} = \sum_{i,j} |i - j|^2 P(i,j)$$
2. **Homojenlik (Homogeneity):** Matris elemanlarının köşegene yakınlığını ölçer:
   $$\text{Homojenlik} = \sum_{i,j} \frac{P(i,j)}{1 + |i - j|}$$
3. **Enerji / Açısal İkinci Moment (Energy / ASM):** Dokunun düzenliliğini ve periyodikliğini ölçer:
   $$\text{Enerji} = \sqrt{\sum_{i,j} P(i,j)^2}$$
4. **Korelasyon (Correlation):** Piksel çiftleri arasındaki doğrusal bağımlılığı ölçer:
   $$\text{Korelasyon} = \sum_{i,j} \frac{(i - \mu_i)(j - \mu_j) P(i,j)}{\sigma_i \sigma_j}$$
5. **Entropi (Entropy):** Dokuma deseninin karmaşıklık ve rastgelelik seviyesini ölçer:
   $$\text{Entropi} = -\sum_{i,j} P(i,j) \log_2(P(i,j) + \epsilon)$$

---

### 3. Katalog Arama Sonuçları (Kamera Çekimi Simülasyonu Sorgusu)

| Sıra | Halı Kodu | Halı Başlığı | Kategori | Hibrit Skor (%) | Renk Skoru (%) | Doku Skoru (%) |
|---|---|---|---|---|---|---|
| **#1** | `CARPET-CLASSIC-01` | **Hereke Klasik Madalyonlu Bordo** | Klasik / Geleneksel | **%98.42** | %99.10 | %97.58 |
| **#2** | `CARPET-VINTAGE-04` | **Anadolu Eskitme Terracotta** | Vintage / Eskitme | **%74.85** | %76.20 | %73.20 |
| **#3** | `CARPET-SILK-05` | **Osmanlı İpek Çiçekli Zümrüt** | Klasik / Saray | **%68.40** | %64.50 | %73.18 |
| **#4** | `CARPET-MODERN-02` | **İskandinav Geometrik Triko** | Modern / Geometrik | **%61.20** | %63.80 | %58.02 |

---

## 🛠️ Dizin Yapısı

```
day-38-carpet-visual-retrieval/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # numpy, scipy, pillow, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca katalog indeksleme ve görsel arama betiği
├── README.md                        # 220+ satır sektörel ve matematiksel dokümantasyon
├── src/
│   ├── __init__.py
│   ├── renk_cikarici.py             # 3B HSV Histogramı ve Renk Momentleri çıkarıcı
│   ├── doku_cikarici.py             # GLCM (Haralick 4 yönlü) ve LBP mikro-doku çıkarıcı
│   ├── fuzyon_arama_motoru.py       # Ağırlıklı çoklu özellik füzyonu ve Top-K arama motoru
│   ├── hali_katalog_verisi.py       # Sentetik halı dokuma desenleri ve katalog kütüphanesi
│   └── gorsellestirici.py           # 6 panelli görsel arama ve teşhis panosu
├── testler/
│   ├── __init__.py
│   └── test_carpet_retrieval.py     # 7 adet birim test (Tümü Başarılı)
└── ciktilar/
    └── hali_gorsel_arama_paneli.png # 6 panelli yüksek çözünürlüklü arama teşhis görseli
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

**Görev:** `src/fuzyon_arama_motoru.py` içerisine **"Kullanıcı Tercihli Arama Modları (Search Modes)"** ekleyerek aramayı 3 farklı moda uyarlamak:
1. `RENK_ODAKLI`: Renk ağırlığı $w_r = 0.85$, Doku ağırlığı $w_d = 0.15$
2. `DOKU_ODAKLI`: Renk ağırlığı $w_r = 0.15$, Doku ağırlığı $w_d = 0.85$
3. `DENGE_MODU`: Renk ağırlığı $w_r = 0.50$, Doku ağırlığı $w_d = 0.50$

**Tamamlanan Çözüm:**
```python
def moda_gore_ara(self, sorgu_gorseli, mod: str = "DENGE_MODU", top_k: int = 3):
    mod_haritasi = {
        "RENK_ODAKLI": 0.85,
        "DOKU_ODAKLI": 0.15,
        "DENGE_MODU": 0.50
    }
    agirlik = mod_haritasi.get(mod, 0.50)
    return self.gorsel_ara(sorgu_gorseli, top_k=top_k, ozel_renk_agirligi=agirlik)
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Halı ve tekstil görsel aramasında GLCM matrisi hesaplanırken neden tek bir yön ($\theta = 0^\circ$) yerine 4 farklı yön ($\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$) kullanılır?

> **Mentor Cevabı:**
> Halı dokuma ve jakar yapısı **anizotropik (yöne bağımlı)** özellik gösterir. Örneğin dikey çözgü iplikleri ile yatay atkı ipliklerinin örgü frekansı veya çizgili/çapraz jakar motiflerinin yönü her açıda farklıdır. Sadece yatay ($0^\circ$) açıya bakıldığında dikey çizgili bir desen homojen görünebilir ve yönsel periyodisite kaybedilir. 4 yönün ortalaması alınarak veya yön vektörleri birleştirilerek **dönme değişmezliği (rotation invariance)** ve tam yönsel doku spektrumu elde edilir.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
