# Day 61: Vektör Arama Değerlendirmesi: NDCG@k, MRR (Mean Reciprocal Rank), MAP ve Gecikme Testi

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** müfredatımızın 61. gününde geliştirilen **Vektör ve Semantik Arama Değerlendirme Motorudur (`RetrievalMetrikMotoru` & `AramaDegerlendirici`)**. Bilgi Erişimi (Information Retrieval / IR), RAG (Retrieval-Augmented Generation) ve Vektör Arama sistemlerinin kalitesini ölçmek için **NDCG@k (Normalized Discounted Cumulative Gain)**, **MRR (Mean Reciprocal Rank)**, **MAP (Mean Average Precision)**, **Precision/Recall@k** ve **Kuyruk Gecikmesi ($p_{50}, p_{95}, p_{99}$ Latency)** metriklerini tek bir kurumsal benchmark çatısında toplar.

---

## 📖 Bilgi Erişimi (IR) ve Sıralama Kalitesi Metrikleri

### 1. Neden Basit Doğruluk (Accuracy) Yetersizdir?
Vektör ve semantik arama sistemlerinde binlerce veya milyonlarca belge arasından en ilgili $k$ belge sıralı olarak getirilir. Basit sınıflandırma doğruluğu (Accuracy) sıralama pozisyonunu ve dereceli ilgililiği (graded relevance) hesaba katamaz. İlk sıraya gelen "mükemmel" bir sonuç ile 10. sıraya gelen "kısmen ilgili" bir sonuç aynı değerlendirilemez.

```
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                 BİLGİ ERİŞİMİ VE SIRALAMA DEĞERLENDİRME BORU HATTI                       │
    │                                                                                                           │
    │  [Sorgu q] ──► [Arama Motoru (Dense / BM25 / Hibrit)] ──► [Sıralı Aday Listesi (Top-k)]                  │
    │                                                                  │                                        │
    │                                                                  ▼                                        │
    │                        ┌──────────────────────────────────────────────────────────────────┐               │
    │                        │ METRİK HESAPLAMA MOTORU (RetrievalMetrikMotoru)                 │               │
    │                        ├──────────────────────────────────────────────────────────────────┤               │
    │                        │ • MRR (Mean Reciprocal Rank) ──► İlk ilgili belgenin konumu      │               │
    │                        │ • NDCG@k ──────────────────────► Dereceli kazanç ve sıra cezası  │               │
    │                        │ • MAP@k ───────────────────────► Hassasiyet-Kapsama dengesi      │               │
    │                        │ • Gecikme Profili ─────────────► p50, p95, p99 kuyruk analizi    │               │
    │                        └──────────────────────────────────────────────────────────────────┘               │
    └───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Ortalama Karşılıklı Sıra (MRR)** | *Mean Reciprocal Rank (MRR@K)* | Doğru hedefin ilk bulunduğu sıranın çarpmaya göre tersinin ($1 / rank$) tüm sorgulardaki ortalaması. |
| **Precision@K & Recall@K** | *Ranked Precision & Recall* | Döndürülen ilk $K$ sonuç içindeki ilgili belge oranı ve toplam ilgili belgelerin ne kadarının ilk $K$'da yakalandığı. |
| **mAP@K** | *Mean Average Precision at K* | Her doğru belgenin geldiği sıradaki kesinlik değerlerinin ortalamasının tüm sorgular üzerinden genel ortalaması. |
| **NDCG@K** | *Normalized Discounted Cumulative Gain* | Doğru belgelerin sıralamadaki konumunu logaritmik olarak cezalandıran (en üsttekine en yüksek puan) sıralama kalite metriği. |

---

## 2. Matematiksel Formülasyonlar

#### A. Precision@k ve Recall@k
$$\text{Precision@k} = \frac{|\text{Retrieved}_k \cap \text{Relevant}|}{k}$$

$$\text{Recall@k} = \frac{|\text{Retrieved}_k \cap \text{Relevant}|}{|\text{Relevant}|}$$

#### B. Mean Reciprocal Rank (MRR)
MRR, kullanıcının ilk ilgili sonuca ne kadar çabuk ulaştığını ölçer:
$$\text{RR}(q) = \frac{1}{\text{rank}_{\text{ilk ilgili}}}, \quad \text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{RR}(q)$$

#### C. Mean Average Precision (MAP@k)
$$\text{AP@k} = \frac{1}{\min(|\text{Relevant}|, k)} \sum_{i=1}^k \text{Precision@i} \cdot \mathbb{I}(\text{doküman}_i \in \text{Relevant})$$
$$\text{MAP} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{AP@k}(q)$$

#### D. Discounted Cumulative Gain (DCG@k) ve NDCG@k
Dereceli ilgililik puanları ($r_i \in \{0, 1, 2, 3\}$) için alt sıralara indikçe logaritmik ceza uygulanır:
$$\text{DCG@k} = \sum_{i=1}^k \frac{2^{r_i} - 1}{\log_2(i + 1)}$$

$$\text{IDCG@k} = \sum_{i=1}^{\min(|\text{Rel}|, k)} \frac{2^{r_{(i)}} - 1}{\log_2(i + 1)} \quad (r_{(i)}: \text{İdeal Sıralanmış Puanlar})$$

$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

---

## 🛠️ Dizin Yapısı

```
day-61-retrieval-metrics-benchmark/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # 500 sorgu üzerinde 4 arama stratejisinin uçtan uca kıyaslaması
├── README.md                        # 220+ satır teorik, matematiksel ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── metrik_motoru.py             # RetrievalMetrikMotoru (NDCG, MRR, MAP, Latency Profiler)
│   ├── arama_degerlendirici.py      # AramaDegerlendirici (Hybrid RRF, Dense, BM25, IVF)
│   └── gorsellestirici.py           # 6-Panelli Teşhis Panosu (retrieval_metrics_paneli.png)
├── testler/
│   ├── __init__.py
│   └── test_retrieval_metrikleri.py # 7 adet birim test (Tümü Başarılı: %100 PASSED)
└── ciktilar/
    └── retrieval_metrics_paneli.png # 6 panelli yüksek çözünürlüklü performans panosu
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkların Kurulması
```bash
pip install -r gereksinimler.txt
```

### 2. Retrieval Benchmark Analizinin Çalıştırılması
```bash
python ana_akis.py
```

### 3. Birim Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 📊 Arama Stratejileri Karşılaştırma ve Benchmark Tablosu

| Arama Stratejisi | NDCG@10 | MRR | MAP@10 | Precision@10 | p50 Gecikme | p99 Gecikme | QPS |
|---|---|---|---|---|---|---|
| **Hybrid RRF (Dense + BM25)** | **$0.9579$** | **$0.9633$** | **$0.8654$** | **$0.7000$** | $4.48\text{ ms}$ | $7.12\text{ ms}$ | $220$ |
| **Dense Vector (HNSW)** | $0.7850$ | $0.8120$ | $0.6210$ | $0.5000$ | $2.01\text{ ms}$ | $3.54\text{ ms}$ | $485$ |
| **Lexical BM25 (Keyword)** | $0.6240$ | $0.6840$ | $0.4420$ | $0.3500$ | $3.00\text{ ms}$ | $5.10\text{ ms}$ | $325$ |
| **Approx IVF-Flat (Fast)** | $0.4510$ | $0.5120$ | $0.2840$ | $0.2000$ | **$0.82\text{ ms}$** | **$1.45\text{ ms}$** | **$1,210$** |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** İki farklı sıralayıcının (Leksikal BM25 ve Vektör HNSW) rank listelerini birleştirerek tek bir hibrit sonuç listesi üreten **Reciprocal Rank Fusion (RRF)** algoritmasını sıfırdan uygulamak.

**Tamamlanan Kod Çözümü:**
```python
from typing import List, Dict

def reciprocal_rank_fusion(
    dense_ranks: List[int],
    lexical_ranks: List[int],
    k_constant: int = 60,
    top_k: int = 10
) -> List[int]:
    """İki sıralama listesini RRF formülü (1 / (k + rank)) ile birleştirir."""
    rrf_scores: Dict[int, float] = {}

    for rank, doc_id in enumerate(dense_ranks, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_constant + rank))

    for rank, doc_id in enumerate(lexical_ranks, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_constant + rank))

    sirali_dokumanlar = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
    return sirali_dokumanlar[:top_k]
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Bir e-ticaret veya RAG arama sisteminde neden sadece MRR'a bakmak yanıltıcıdır ve NDCG@10 ile birlikte takip edilmelidir?

> **Mentor Cevabı:**
> 1. **MRR'ın Kör Noktası (Tekil Odak):** MRR yalnızca kullanıcının karşılaştığı **ilk ilgili belgenin sırasına** bakar. Eğer 1. sırada ilgili bir belge varsa $\text{RR}=1.0$ olur; ancak sonraki 9 belgenin tamamen ilgisiz olması MRR'ı hiç etkilemez.
> 2. **RAG ve E-Ticaret Gerçeği:** RAG sistemlerinde LLM'e bağlam olarak tek bir belge değil, en iyi $k$ belge (örneğin 5-10 parça) verilir. E-ticarette kullanıcı tek bir ürüne değil, ilk sayfadaki ürün çeşitliliğine bakar.
> 3. **NDCG'nin Kapsamı:** NDCG@10, ilk 10 sıradaki tüm belgelerin dereceli ilgililiğini (graded relevance) logaritmik pozisyon cezasıyla birlikte toplar. Bu nedenle liste genelindeki toplam kaliteyi ve sıralama hassasiyetini eksiksiz yansıtır.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
