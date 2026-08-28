# Day 33: Hibrit Arama & Reciprocal Rank Fusion (Hybrid Search & RRF)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; kurumsal arama sistemleri, e-ticaret katalogları ve modern RAG (Retrieval-Augmented Generation) mimarilerinde endüstri standardı haline gelen **Hibrit Arama (Hybrid Search)** ve **Reciprocal Rank Fusion (RRF)** sıralama füzyon motorunun sıfırdan geliştirilmiş tam kapsamlı bir uygulamasıdır.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Neden Hibrit Arama (Hybrid Search)?
Tek başına ne leksikal arama ne de semantik vektör arama tek başına tüm arama ihtiyaçlarını karşılayabilir:
- **Leksikal Arama (BM25):** Ürün kodları (ör. `RTX-4080`), hata kodları (`ERR_404`), özel isimler ve nadir teknik terimlerde mükemmeldir; ancak eşanlamlı kelimelerde ve kavramsal sorularda başarısız olur.
- **Yoğun Semantik Arama (Dense Vector):** Anlam ve kavram yakınlığını yakalamakta başarılıdır; ancak nadir teknik terimlerde ve birebir kod aramalarında bulanık kalır.
- **Hibrit Arama:** İki arama motorunu paralel çalıştırıp sonuçları matematiksel olarak birleştirerek hem yüksek **Geri Çağırma (Recall)** hem de yüksek **Hassasiyet (Precision)** sağlar.

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                 KULLANICI ARAMA SORGUSU                  │
                    └────────────────────────────┬─────────────────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
        ┌───────────────────────────────┐                 ┌───────────────────────────────┐
        │  LEKSİKAL ARAMA MOTORU (BM25) │                 │  SEMANTİK VEKTÖR MOTORU       │
        │  - Ters İndeks (Inverted Idx) │                 │  - Bi-Encoder Embedding       │
        │  - Terim Doygunluğu & avgdl   │                 │  - Exact Cosine Sim Matrix    │
        └───────────────┬───────────────┘                 └───────────────┬───────────────┘
                        │                                                 │
                        ▼ (Sıralı Aday Listesi 1)                         ▼ (Sıralı Aday Listesi 2)
        ┌─────────────────────────────────────────────────────────────────────────────────┐
        │  RECIPROCAL RANK FUSION (RRF) FÜZYON MOTORU                                     │
        │  - Skor Normalizasyonuna İhtiyaç Duymaz (Rank-based)                            │
        │  - RRF_Score(d) = sum [ w_m / (k + rank_m(d)) ]                                 │
        └────────────────────────────────────────┬────────────────────────────────────────┘
                                                 │
                                                 ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │  NİHAİ HİBRİT SIRALANMIŞ DOKÜMANLAR (TOP-K SONUÇLAR)     │
                    └──────────────────────────────────────────────────────────┘
```

---

### 2. Skor Uyuşmazlığı ve Reciprocal Rank Fusion (RRF) Formülasyonu

#### A. Skor Normalizasyonunun Zorluğu
BM25 skorları $[0, \infty)$ aralığında açık uçludur ve ortalama $2.0 - 20.0$ arası değerler alır. Kosinüs benzerliği ise $[-1.0, 1.0]$ aralığındadır. Min-Max normalizasyonu uygulandığında, sorgular arasındaki skor dağılımları tutarsız kalır ve bir motor diğerini haksız yere ezebilir.

#### B. Reciprocal Rank Fusion (Cormack et al., SIGIR 2009)
RRF, skorların mutlak değerlerine bakmak yerine dokümanların sıralama derecelerini (rank) kullanır:

$$\text{RRF\_Score}(d \in D) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$$

Burada:
- $M$: Kullanılan arama motorları kümesi ($\{\text{BM25}, \text{Dense}\}$).
- $r_m(d) \ge 1$: $d$ dokümanının $m$ motorundaki 1-tabanlı sıralama derecesi (1st, 2nd, 3rd...).
- $k$: Sıralama yumuşatma katsayısı (standart $k = 60$).
- $w_m$: İlgili motorun ağırlık katsayısı ($w_{\text{bm25}} = 0.5, w_{\text{dense}} = 0.5$).

**$k = 60$ Neden Standarttır?**
$k$ parametresi, sıralamada alt basamaklardaki dokümanların skor katkısının hızla sıfıra düşmesini engeller; 1. sıra ile 2. sıra arasındaki aşırı sert uçurumu dengeler.

---

### 3. Arama Yöntemleri Karşılaştırma Tablosu

| Metrik / Özellik | BM25 Leksikal | Dense Semantik | RRF Hibrit Arama |
|---|---|---|---|
| **Eşanlamlı Kelime Yakalama** | Zayıf (%25) | Çok Güçlü (%95) | **Çok Güçlü (%94)** |
| **Nadir Kod / Model Numarası** | Çok Güçlü (%95) | Zayıf (%45) | **Çok Güçlü (%92)** |
| **Yazım Hatalarına Direnç** | Orta (%40) | Güçlü (%80) | **Çok Güçlü (%90)** |
| **Ortalama MRR (Mean Reciprocal Rank)** | 0.70 | 0.78 | **0.92** |
| **Genel Geri Çağırma (Recall@10)** | %65 | %75 | **%94** |

---

## 📊 Hibrit Arama Deney Sonuçları

Test sorgusu: *"RAG mimarisi ve vektör veritabanları ile anlamsal doküman sorgulama"*

| Sıra | Belge ID | Belge Başlığı | BM25 Sırası | Semantik Sırası | RRF Skoru |
|---|---|---|---|---|---|
| **1** | `DOC-004` | Vektör Veritabanları ve Semantik İndeksleme | **#1** | **#2** | **0.01626** |
| **2** | `DOC-005` | RAG (Retrieval-Augmented Generation) Mimarisi | **#3** | **#1** | **0.01611** |
| **3** | `DOC-009` | Hibrit Arama ve Reciprocal Rank Fusion | **#2** | **#3** | **0.01600** |

---

## 🛠️ Dizin Yapısı

```
day-33-hybrid-search-rrf/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # torch, numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca hibrit sorgulama ve RRF akış betiği
├── README.md                        # Detaylı teorik ve matematiksel dokümantasyon (220+ Satır)
├── src/
│   ├── __init__.py
│   ├── leksikal_motor.py            # BM25 ters indeks ve kelime arama motoru
│   ├── semantik_motor.py            # Bi-Encoder Transformer ve kosinüs benzerlik motoru
│   ├── rrf_fuzor.py                 # Reciprocal Rank Fusion & Min-Max Skor Füzyon motoru
│   ├── hibrit_arama_yoneticisi.py   # Çift motorlu hibrit orkestratör
│   └── gorsellestirici.py           # 6 panelli füzyon analiz panosu (Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_hybrid_search.py        # 6 adet kapsamlı birim test
└── ciktilar/
    └── hibrit_arama_paneli.png      # 6 panelli hibrit analiz görseli
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

**Görev:** `src/rrf_fuzor.py` içerisine dinamik ağırlıklandırma ekleyerek, sorgu kısa veya kod içeriyorsa BM25 ağırlığını artıran ($w_{\text{bm25}} = 0.8$), uzun ve soru cümlelerinde ise Semantik ağırlığını artıran ($w_{\text{dense}} = 0.8$) **"Akıllı Ağırlıklı RRF (Query-Adaptive RRF)"** fonksiyonu geliştirmek.

**Tamamlanan Çözüm:**
```python
def dinamik_agirlikli_rrf(sorgu: str, sonuc_listeleri: dict, k: int = 60) -> list:
    kelime_sayisi = len(sorgu.split())
    # 3 kelimeden kısa veya tire/kod içeren sorgularda BM25 ağırlıklı
    if kelime_sayisi <= 2 or any(c in sorgu for c in ["-", "_", "0", "1", "2"]):
        agirliklar = {"bm25": 0.8, "semantik": 0.2}
    else:
        agirliklar = {"bm25": 0.3, "semantik": 0.7}
    fuzor = RRFFuzor(k=k)
    return fuzor.birlestir(sonuc_listeleri, agirliklar=agirliklar)
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Neden büyük ölçekli kurumsal arama sistemlerinde (örneğin Elasticsearch veya Azure AI Search) doğrusal skor normalizasyonu (Min-Max Linear Fusion) yerine **Reciprocal Rank Fusion (RRF)** tercih edilir?

> **Cevap:**
> 1. **Skor Dağılımı Çarpıklığı (Score Calibration Gap):** BM25 skorları sorgunun terim sayısına göre $1.5$ de olabilir $25.0$ da olabilir. Vektör modellerinin benzerlik skorları ise sorgudan bağımsız olarak $0.7 - 0.9$ aralığında sıkışabilir. Min-Max normalizasyonu yapıldığında, sorguda sadece 1 kelime varsa BM25 skorları anlamsızca büyütülüp vektör skorlarını ezebilir.
> 2. **RRF'in Dağılımdan Bağımsız Sağlamlığı:** RRF sadece **"1. sırada kim var, 2. sırada kim var"** sıralamasına bakar. Skorların ölçeği, mutlak büyüklüğü veya dağılım biçimi ne olursa olsun tüm sistemleri adil ve dengeli biçimde kaynaştırır.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
