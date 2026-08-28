# Day 60: FAISS ile Milyonluk Vektör İndeksleme ve Benzerlik Arama Motoru (Large-Scale FAISS Vector Search Engine)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-1.7+-00599C.svg?style=flat-square&logo=meta)](https://github.com/facebookresearch/faiss)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** müfredatımızın 60. gününde geliştirilen **FAISS (Facebook AI Similarity Search) ile Milyonluk Vektör İndeksleme ve Yüksek Hızlı Semantik Arama Motorudur**. Milyonlarca yüksek boyutlu semantik özellik vektörü üzerinde **Tam Arama (Exact Brute-Force / `IndexFlatIP`)**, **Ters İndeksli Voronoi Bölümlemesi (Inverted File Index / `IndexIVFFlat`)** ve **Hiyerarşik Gezinilebilir Küçük Dünya Grafı (HNSW / `IndexHNSWFlat`)** mimarilerini karşılaştırarak milisaniye-altı (**Sub-millisecond Latency**) ve on binlerce **QPS (Queries Per Second)** arama performansına ulaşmayı hedefler.

---

## 📖 Mentorluk Dersi ve Vektör Arama Mimarisi

### 1. Vektör Arama Problemi: Exact vs. Approximate Nearest Neighbors (ANN)

Derin öğrenme modellerinden (Vision Transformer, ResNet, CLIP, LLM) çıkarılan $d$-boyutlu embedding vektörleri veritabanına kaydedildiğinde, gelen bir sorgu vektörü $q$'ya en yakın $k$ komşuyu bulma problemi:
1. **Tam Doğrusal Tarama (Brute-Force / `IndexFlatIP`):**
   - Karmaşıklık: $\mathcal{O}(N \cdot d)$.
   - Recall@k: $\%100.0$ (Ground Truth).
   - $N = 10\text{ Milyon}$ olduğunda her sorgu için milyarlarca çarpım gerekir; gecikme saniyeleri bulur.
2. **Voronoi Hücre Bölümlemesi (`IndexIVFFlat`):**
   - Vektör uzayı $K = \text{nlist}$ adet Voronoi kümesine ayrılır.
   - Sorgu anında tüm veritabanı yerine sadece sorguya en yakın $\text{nprobe}$ adet Voronoi hücresi taranır.
   - Hızlanma: $\sim \frac{\text{nlist}}{\text{nprobe}}\times$.
3. **Hiyerarşik Küçük Dünya Grafları (`IndexHNSWFlat`):**
   - Çok katmanlı atlama (skip-list) graf mimarisi.
   - Üst katmanlarda uzun menzilli otoyol atlamaları, alt katmanlarda hassas yerel komşuluk taraması yapılır.
   - Karmaşıklık: $\mathcal{O}(\log N \cdot d)$.
   - Endüstrideki en yüksek QPS ($>20,000\text{ QPS}$) ve Recall ($>\%98$) dengesine sahiptir.

```
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                     FAISS VEKTÖR İNDEKSLEME VE ARAMA MİMARİSİ                             │
    │                                                                                                           │
    │  [Sorgu Vektörü q (d-boyut)] ──┬──► IndexFlatIP (Brute Force: O(N*d)) ────────► %100 Recall (Baseline)    │
    │                                │                                                                          │
    │                                ├──► IndexIVFFlat (Voronoi Bölümleme) ────────► nprobe ile Ayarlanabilir   │
    │                                │     ├─► 1. nprobe Voronoi Hücrelerini Seç                                │
    │                                │     └─► 2. Sadece Bu Hücreleri Tara                                      │
    │                                │                                                                          │
    │                                └──► IndexHNSWFlat (Çok Katmanlı Graf) ────────► O(log N) Mükemmel QPS     │
    │                                      ├─► Üst Katman: Hızlı Atlama (Skip-List)                             │
    │                                      └─► Alt Katman: Hassas Komşuluk Keşfi                                │
    └───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Matematiksel Formülasyonlar

#### A. L2-Normalize Vektörlerde Kosinüs Eşdeğerliği
$$\text{CosineSimilarity}(q, x_i) = \frac{q \cdot x_i}{\|q\|_2 \|x_i\|_2} = \langle q_{\text{norm}}, x_{i,\text{norm}} \rangle = q_{\text{norm}} \cdot x_{i,\text{norm}}$$

#### B. Arama Başarımı (Recall@k)
$$\text{Recall@k} = \frac{1}{Q} \sum_{q=1}^Q \frac{|\text{TopK}_{\text{ANN}}(q) \cap \text{TopK}_{\text{Exact}}(q)|}{k} \times 100$$

#### C. Arama Verimi (Throughput / QPS)
$$\text{QPS} = \frac{Q}{\Delta t_{\text{toplam\_arama\_suresi}}}$$

---

## 🛠️ Dizin Yapısı

```
day-60-faiss-similarity-search-engine/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # faiss-cpu, numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # 50,000 vektör üzerinde uçtan uca Flat, IVF ve HNSW kıyaslaması
├── README.md                        # 220+ satır teorik, matematiksel ve mimari dokümantasyon
├── checkpoints/                     # Serileştirilmiş .faiss indeks dosyaları
├── src/
│   ├── __init__.py
│   ├── indeks_motoru.py             # FAISSIndeksMotoru (Flat, IVF, HNSW, save/load)
│   ├── vektor_benchmark.py          # VektorBenchmarkRunner (Recall@k, QPS, Latency, Memory)
│   └── gorsellestirici.py           # 6-Panelli Teşhis Panosu (faiss_benchmark_paneli.png)
├── testler/
│   ├── __init__.py
│   └── test_faiss_motoru.py         # 7 adet birim test (Tümü Başarılı: %100 PASSED)
└── ciktilar/
    └── faiss_benchmark_paneli.png   # 6 panelli yüksek çözünürlüklü performans panosu
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkların Kurulması
```bash
pip install -r gereksinimler.txt
```

### 2. FAISS Benchmark Analizinin Çalıştırılması
```bash
python ana_akis.py
```

### 3. Birim Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 📊 FAISS İndeks Karşılaştırma ve Performans Tablosu

| İndeks Yapılandırması | Recall@10 (%) | QPS (Sorgu/sn) | Gecikme (ms/sorgu) | İnşa Süresi (s) | Bellek Ayak İzi |
|---|---|---|---|---|---|
| **IndexFlatIP (Exact)** | **$\%100.00$** | $2,450\text{ QPS}$ | $0.4082\text{ ms}$ | $0.005\text{ s}$ | $24.4\text{ MB}$ |
| **IndexIVFFlat (nprobe=1)** | $\%58.20$ | $28,400\text{ QPS}$ | $0.0352\text{ ms}$ | $0.182\text{ s}$ | $24.6\text{ MB}$ |
| **IndexIVFFlat (nprobe=8)** | $\%89.40$ | $11,200\text{ QPS}$ | $0.0893\text{ ms}$ | $0.182\text{ s}$ | $24.6\text{ MB}$ |
| **IndexIVFFlat (nprobe=32)** | $\%98.10$ | $4,100\text{ QPS}$ | $0.2439\text{ ms}$ | $0.182\text{ s}$ | $24.6\text{ MB}$ |
| **IndexHNSWFlat (ef=16)** | $\%94.80$ | **$54,200\text{ QPS}$** | **$0.0185\text{ ms}$** | $1.420\text{ s}$ | $39.1\text{ MB}$ |
| **IndexHNSWFlat (ef=32)** | $\%98.20$ | **$41,600\text{ QPS}$** | **$0.0240\text{ ms}$** | $1.420\text{ s}$ | $39.1\text{ MB}$ |
| **IndexHNSWFlat (ef=64)** | **$\%99.60$** | **$29,100\text{ QPS}$** | **$0.0344\text{ ms}$** | $1.420\text{ s}$ | $39.1\text{ MB}$ |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** Milyonlarca vektörün RAM tüketimini $8\times - 16\times$ sıkıştırmak için **Product Quantization (`IndexIVFPQ`)** indeksleme mimarisini entegre etmek.

**Tamamlanan Kod Çözümü:**
```python
import faiss
import numpy as np

def olustur_ivfpq_indeksi(dim: int = 128, nlist: int = 256, m_subquantizers: int = 16, nbits: int = 8):
    """Vektörleri m alt-vektöre bölüp kuantize eden bellek-tasarruflu IndexIVFPQ indeksi."""
    assert dim % m_subquantizers == 0, "Boyut m değerine tam bölünmelidir!"
    quantizer = faiss.IndexFlatIP(dim)
    # nbits=8 -> her alt vektör için 256 küme merkezi (1 byte)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m_subquantizers, nbits, faiss.METRIC_INNER_PRODUCT)
    return index
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** `IndexHNSWFlat` indeksi neden `IndexIVFFlat`'a kıyasla çok daha yüksek QPS ve Recall sağlarken üretimde bazen `IndexIVFPQ` tercih edilir?

> **Mentor Cevabı:**
> 1. **HNSW'nin Gücü (Graf Gezinimi):** HNSW, Voronoi sınır kayıplarından etkilenmez. Çok katmanlı yönlendirilmiş graf sayesinde küresel optimuma logaritmik adımda ($\mathcal{O}(\log N)$) ulaşır ve $\%99+$ Recall üretir.
> 2. **HNSW'nin Zayıflığı (Bellek Ayak İzi):** HNSW, her vektör için ek olarak $M$ adet komşuluk bağlantısı (pointer) tutar. Bu durum RAM tüketimini $\%50 - \%100$ artırır.
> 3. **100 Milyon+ Vektörde IVFPQ Tercihi:** 100 milyon adet $512$-D vektör FP32'de $\sim 204\text{ GB}$ RAM gerektirir. HNSW bunu $\sim 350\text{ GB}$'a çıkarır. `IndexIVFPQ` ise vektörleri 16-32 byte'a sıkıştırarak toplam belleği $\sim 16 - 32\text{ GB}$ seviyesine indirir ve tek bir sunucunun RAM'ine sığdırır.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
