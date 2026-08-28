# Day 32: Yoğun Vektör Tabanlı Semantik Arama Motoru (Dense Semantic Search Engine)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; modern yapay zeka, Vektör Veritabanları (FAISS, Pinecone, Qdrant, Chroma) ve RAG (Retrieval-Augmented Generation) sistemlerinin kalbinde yer alan **Yoğun Vektör Tabanlı Semantik Arama Motorunun (Dense Semantic Search Engine)**; **Bi-Encoder Transformer**, **Mean Pooling**, **L2 Normalizasyonu** ve **Exact Cosine $k$-NN Vektör İndeksi** bileşenleriyle sıfırdan geliştirilmiş tam kapsamlı bir uygulamasıdır.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Neden Leksikal Arama (BM25) Yetersiz Kalır?
Geleneksel anahtar kelime eşleştirme yöntemleri (BM25, TF-IDF), metinlerin **yüzey formlarına (surface form)** bağımlıdır:
- **Eşanlamlılık (Synonymy) Problemi:** Kullanıcı `"kalp spazmı ve göğüs sıkışması"` aradığında, veritabanında `"miyokard enfarktüsü kardiyak semptomları"` yazıyorsa leksikal motorlar ortak kelime bulamadığı için **0 puan** üretir.
- **Kelimelerin Anlamsal Yakınlığı:** Cümle Kodlayıcıları (Sentence Transformers), metinleri yüzlerce boyutlu sürekli bir vektör uzayına ($\mathbb{R}^D$) eşler; böylece aynı anlama gelen farklı ifadeler geometrik olarak birbirine yakın kümelenir.

```
                    ┌──────────────────────────────────────────────────────────┐
                    │            DOKÜMAN METNİ / KULLANICI SORĞUSU             │
                    └────────────────────────────┬─────────────────────────────┘
                                                 │
                                                 ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │  BI-ENCODER TRANSFORMER KODLAYICI                        │
                    │  - Token Embeddings + Positional Encodings               │
                    │  - Multi-Head Self-Attention Katmanları                  │
                    └────────────────────────────┬─────────────────────────────┘
                                                 │
                                                 ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │  ATTENTION MASK WEIGHTED MEAN POOLING                    │
                    │  - Token tensörlerini tek bir cümle vektörüne indirgeme  │
                    └────────────────────────────┬─────────────────────────────┘
                                                 │
                                                 ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │  L2 NORMALİZASYONU (BİRİM KÜRE PROJEKSİYONU)             │
                    │  - ||v||_2 = 1.0 (Skaler çarpım = Kosinüs Benzerliği)   │
                    └────────────────────────────┬─────────────────────────────┘
                                                 │
                                                 ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │  DÜZ VEKTÖR İNDEKSİ (FLAT MATRIX MULTIPLICATION)         │
                    │  - s = V * q^T -> En yüksek kosinüs benzerliğine sahip   │
                    │    Top-k dokümanların çıkarılması                        │
                    └──────────────────────────────────────────────────────────┘
```

---

#

---

### 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Metinleri yüksek boyutlu anlamsal embedding vektörlerine dönüştürerek kosinüs benzerliği ile anlamsal yakınlığa göre arama yapmak için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Kullanıcı farklı kelimeler kullansa dahi (ör. 'araç' ve 'otomobil') aynı anlama gelen belgelerin bulunmasını sağlar.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Nadir geçen özel kodlarda, marka isimlerinde ve sayısal değerlerde BM25 kadar keskin eşleşme yapamayabilir.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- BM25, ColBERT, Cross-Encoder Re-ranker veya Hibrit Vektör Arama.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Yoğun Vektör Temsili** | *Dense Vector Embedding* | Metinlerin anlamsal içeriğini 384 veya 768 boyutlu sürekli uzayda temsil eden öğrenilmiş vektörler. |
| **Sentence Transformers** | *Sentence-BERT (SBERT)* | Cümle ve paragrafları siyam ağı mimarisiyle anlamsal vektör uzayına haritalayan transformer modelleri. |
| **Biy-Kodlayıcı (Bi-Encoder)** | *Bi-Encoder Architecture* | Sorgu ve belgeleri birbirinden bağımsız olarak vektörleştirip kosinüs benzerliği ile hızlı arama sağlayan mimari. |
| **Anlamsal Yakınlık** | *Semantic Proximity* | Kelimeler birebir eşleşmese dahi (ör. 'otomobil' ve 'araç') anlam yakınlığı üzerinden doğru belgeleri getirebilme yeteneği. |

---

## 2. Matematiksel Formülasyon

#### A. Ağırlıklı Ortalama Havuzlama (Attention Mask Weighted Mean Pooling)
Bir $L$ uzunluğundaki cümle için Transformer modelinin ürettiği token gizli durumları $\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_L \in \mathbb{R}^D$ ve attention maskesi $m_1, m_2, \dots, m_L \in \{0, 1\}$ olsun:

$$\mathbf{u} = \frac{\sum_{i=1}^L m_i \cdot \mathbf{h}_i}{\sum_{i=1}^L m_i + \epsilon}$$

#### B. L2 Normalizasyonu (Birim Vektör)
Vektörler birim küreye normalize edilir:

$$\mathbf{e} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2} = \frac{\mathbf{u}}{\sqrt{\sum_{j=1}^D u_j^2} + \epsilon}$$

#### C. Kosinüs Benzerliği ve Matris Çarpımı
İki normalize edilmiş vektör $\mathbf{q}$ ve $\mathbf{d}$ arasındaki kosinüs benzerliği, doğrudan skaler çarpımlarına eşittir:

$$\cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2} = \mathbf{q} \cdot \mathbf{d} = \sum_{j=1}^D q_j \cdot d_j$$

İndekste $N$ adet doküman vektörü $\mathbf{V} \in \mathbb{R}^{N \times D}$ saklandığında, tüm dokümanların benzerlik skorları tek bir BLAS matris-vektör çarpımıyla hesaplanır:

$$\mathbf{s} = \mathbf{V} \mathbf{q}^T \in \mathbb{R}^N$$

---

### 3. Bi-Encoder vs Cross-Encoder Karşılaştırması

| Özellik | Bi-Encoder (Bu Proje) | Cross-Encoder |
|---|---|---|
| **Mimari** | Sorgu ve Doküman birbirinden bağımsız kodlanır | Sorgu ve Doküman tek bir girdi olarak birleştirilir (`[CLS] Q [SEP] D`) |
| **Önceden İndeksleme** | Doküman vektörleri 1 kez hesaplanıp veritabanında saklanır | Mümkün değildir; her sorgu için $N$ kez Transformer çalıştırılır |
| **Arama Gecikmesi (Latency)** | **< 1 ms** (Milisaniye altı matris çarpımı) | **~1000 ms** (Ağır hesaplama maliyeti) |
| **Kullanım Alanı** | Milyonlarca doküman arasından ilk Top-100'ü getirme (Retrieval) | Top-100 adayı yüksek doğrulukla yeniden sıralama (Reranking) |

---

## 📊 Semantik Arama Sonuçları

Örnek 10 teknik doküman üzerinde kelime eşleşmesi içermeyen anlamsal sorgu testleri:

| Arama Sorgusu | En İyi Eşleşen Belge | Kosinüs Skoru | Eşleşen Kategori |
|---|---|---|---|
| `"akıllı doküman soru cevaplama ve dil modeli"` | `DOC-005: RAG Asistanı ve LLM Doküman Entegrasyonu` | **0.8842** | Doğal Dil İşleme |
| `"kamera karelerinde hedef öğe koordinat tespiti"` | `DOC-002: Gerçek Zamanlı Nesne Konumlandırma ve YOLO` | **0.8615** | Görüntü İşleme |
| `"yüksek boyutlu embedding benzerlik araması"` | `DOC-004: Vektör Veritabanları ve Benzerlik Araması` | **0.9120** | Veri Mimarisi |

---

## 🛠️ Dizin Yapısı

```
day-32-semantic-search-engine/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # torch, numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca semantik indeksleme ve sorgu yürütme betiği
├── README.md                        # Detaylı teorik ve matematiksel dokümantasyon (220+ Satır)
├── src/
│   ├── __init__.py
│   ├── vektorlestirici.py           # Bi-Encoder Transformer, Mean Pooling ve L2 Normalizasyonu
│   ├── vektor_indeksi.py            # Düz vektör indeksi, exact cosine k-NN ve kategori filtreleme
│   ├── semantik_arama_motoru.py     # Üst seviye orkestratör, PCA 2D izdüşüm motoru
│   └── gorsellestirici.py           # 6 panelli teşhis ve vektör uzayı panosu
├── testler/
│   ├── __init__.py
│   └── test_semantic_search.py      # 6 adet kapsamlı birim test
└── ciktilar/
    └── semantik_arama_paneli.png    # 6 panelli semantik analiz görseli
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

**Görev:** `src/vektor_indeksi.py` içerisine belirli bir kosinüs benzerliği eşiğinin (`min_similarity_threshold = 0.65`) altında kalan alakasız dokümanları filtreleyen bir güven eşiği mekanizması eklemek.

**Tamamlanan Çözüm:**
```python
def esikli_komsu_ara(self, sorgu_vektoru: np.ndarray, top_k: int = 5, min_esik: float = 0.65) -> List[Dict[str, Any]]:
    ham_sonuclar = self.en_yakin_komsu_ara(sorgu_vektoru, top_k=top_k)
    return [r for r in ham_sonuclar if r["skor"] >= min_esik]
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Sentence Transformers modellerinde cümle embedding'i çıkarırken neden `[CLS]` token çıktısını doğrudan almak yerine **Attention Mask Ağırlıklı Mean Pooling** yöntemi tercih edilir?

> **Cevap:**
> 1. **BERT'in Ön Eğitim Hedefi (NSP & MLM):** Standart BERT modellerinde `[CLS]` token'ı genellikle Next Sentence Prediction (NSP) veya sınıflandırma için optimize edilir; tüm cümlenin zengin anlamsal özetini tek başına temsil etmekte zayıf kalır.
> 2. **Mean Pooling'in Üstünlüğü:** Cümledeki tüm token'ların gizli durumlarının (`hidden states`) attention maskesiyle filtrelenerek ortalamasının alınması, cümledeki her bir kelimenin anlamsal katkısını dengeli biçimde birleştirir (Reimers & Gurevych, EMNLP 2019). Yapılan ampirik deneyler, Mean Pooling'in semantik benzerlik (STS) görevlerinde `[CLS]` token'a kıyasla $\%5$-$\%10$ daha yüksek Spearman korelasyonu sağladığını kanıtlamıştır.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
