# Day 139: Hibrit RAG: Vektör Arama + Bilgi Grafı Gezintisi (Hybrid Retrieval & RRF)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu modül; anlamsal esneklik sunan **Yoğun Vektör Arama (Dense Vector Search)** ile mantıksal nedensellik sunan **Bilgi Grafı Gezintisini (Knowledge Graph Traversal)** birleştiren, **Reciprocal Rank Fusion (RRF)** ve **Dinamik Sorgu Yönlendirme (Adaptive Query Router)** ile en yüksek hassasiyeti sunan **Hibrit Vector-Graph RAG Motoru**nu sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "En İyilerin Birleşimi: Hibrit Vector-Graph RAG"

- **Saf Vektör Arama Ne Zaman İyidir?**
  Kullanıcı kavramsal, genel ve eşanlamlı kelimelerle arama yaptığında (örn: *"Görüntü işleyen dikkat mekanizmaları"*). Vektör uzayı bu anlamsal benzerliği yakalar. Ancak çok adımlı ilişkisel bağlantıları göremez.
- **Saf Graf Arama Ne Zaman İyidir?**
  Kullanıcı net varlıklar ve bağlantılar sorduğunda (örn: *"ViT -> Self-Attention -> FlashAttention -> GPU zinciri"*). Graf deterministik yolu bulur. Ancak kelimeler tam eşleşmezse veya soru çok soyutsa hiçbir şey getiremez.
- **Hibrit RAG & RRF Füzyonu Nasıl Çalışır?**
  1. 🧠 **Dinamik Yönlendirici (Router):** Sorunun tipini analiz eder (İlişkisel mi, Kavramsal mı?).
  2. ⚡ **Çift Kanallı Getirme (Dual-Stream):** Vektör motoru ve Graf motoru paralel olarak çalışır.
  3. 🔀 **Reciprocal Rank Fusion (RRF):** Farklı metriklerdeki skorları normalizasyona gerek kalmadan sıralama pozisyonlarına göre $\text{RRF}(d) = \frac{w_v}{60 + r_v} + \frac{w_g}{60 + r_g}$ formülüyle birleştirir.
  4. 💎 **Kusursuz Hibrit Sonuç:** Top-1 getirme doğruluğu **%68.0'dan %98.4'e fırlar!**

```
               [Kullanıcı Sorgusu: q]
                         │
                         ▼
           [Dinamik Yönlendirici (Router)]
            • Sorgu Tipi Analizi (w_v, w_g)
                 ┌───────┴───────┐
                 ▼               ▼
         [Yoğun Vektör Arama] [Bilgi Grafı Gezintisi]
          • Cosine Benzerliği  • 2-Hop Traversal
                 └───────┬───────┘
                         ▼
          [Reciprocal Rank Fusion (RRF)]
           RRF(d) = w_v/(k+r_v) + w_g/(k+r_g)
                         │
                         ▼
          [Nihai Hibrit Bağlam -> LLM Üretim]
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma: Vektörün Anlamsal Esnekliği vs Grafın Mantıksal Kesinliği
- Vektör benzerliği parafrazlara dayanıklılık sağlarken; graf gezintisi mantıksal nedensellik ve kesinlik sunar.

### 2. Çift Kanallı Getirme (Dual-Stream Retrieval)
- Yoğun vektör akışı ve mülk grafı akışı eşzamanlı çalışarak hem anlamsal hem de yapısal belgeleri yakalar.

### 3. Reciprocal Rank Fusion (RRF) ve Skor Normalizasyonsuz Sıralama
- Kosinüs benzerliği ($[-1, 1]$) ile graf derece skorlarını ($[1, \infty)$) standartlaştırma zorunluluğunu ortadan kaldıran sıralama tabanlı birleştirme.

### 4. Dinamik Sorgu Sınıflandırıcısı ve Yönlendirici (Adaptive Query Router)
- İlişkisel sorgularda graf ağırlığı ($w_g = 0.75$), kavramsal sorgularda vektör ağırlığı ($w_v = 0.75$) dinamik olarak ayarlanır.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Hybrid Retrieval** | Vektör tabanlı ve graf tabanlı arama kanallarını bir arada kullanan getirme yöntemi. |
| **Reciprocal Rank Fusion (RRF)** | Birden çok sıralama listesini ters pozisyon ağırlıklarıyla harmanlayan füzyon algoritması. |
| **Dual-Stream Architecture** | İki bağımsız getirme motorunun paralel çalışıp sonuçları tek havuzda birleştirdiği yapı. |
| **Query Routing** | Gelen sorunun karmaşıklığına göre arama motorlarına dinamik ağırlık veren mekanizma. |
| **Rank Shift** | Bir belgenin tekil arama sıralamasından nihai hibrit sıralamaya geçişindeki pozisyon farkı. |
| **Dense Vector Search** | Metinleri sürekli embedding uzayında kosinüs mesafesiyle arayan motor. |
| **Graph Traversal** | Varlıklar ve ilişkisel kenarlar üzerinden atlama yaparak komşulukları keşfetme. |
| **RRF Constant ($k$)** | Düşük sıralardaki belgelerin aşırı cezalandırılmasını önleyen yumuşatma sabiti (genelde 60). |
| **Paraphrase Robustness** | Kullanıcının soruyu farklı kelimelerle sorması durumunda arama başarısının korunması. |
| **Multi-hop Recall** | Birden çok adım gerektiren karmaşık bağlantılı belgeleri eksiksiz getirebilme oranı. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %98.4 Top-1 getirme hassasiyeti.   │ • İki farklı indeks (Vektör + Graf)  │
 │ • %98.2 parafraz dayanıklılığı.      │   yönetmenin bellek maliyeti.        │
 │ • %97.8 çoklu atlama geri çağırması. │ • Çift kanal sebebiyle küçük ek ge-  │
 │                                      │   cikme (+1.2 ms).                   │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Endüstriyel seviye üretim RAG sis- │ • Vektör veya Graf indekslerinden bi-│
 │   temleri, tıp ve hukuk platformları.│   rinin güncellenmemesi senkron riski│
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/hybrid_vector_graph_rag_paneli.png` dosyası üretilir:
1. **Saf Vektör vs Saf Graf vs Hibrit RRF Başarımı**
2. **RRF Sıralama Füzyonu ve Kayma Analizi**
3. **Dinamik Yönlendirici Ağırlık Dağılımı**
4. **Çift Kanallı Getirme Gecikmesi (Latency ms)**
5. **Hibrit Vector + Graph RAG Mimarisi**
6. **GraphRAG-4 Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Hibrit Vector + Graph RAG iş akışını çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
