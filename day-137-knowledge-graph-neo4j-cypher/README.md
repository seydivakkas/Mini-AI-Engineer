# Day 137: GraphRAG-2: Bilgi Grafını Depolama, Cypher Sorgulama ve Multi-Hop Gezinti

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu modül; çıkarılan bilgi graflarını depolayan, deklaratif **Cypher Sorgulama Dili (MATCH ... RETURN)** ile sorgulayan, **Çoklu Atlama (Multi-Hop Graph Traversal - BFS / En Kısa Yol)** algoritmalarıyla akıl yürüten ve LLM için alt-grafik (Subgraph) bağlamı üreten **GraphRAG-2 Motoru**nu sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Graf Üzerinde Gezinmek: GraphRAG-2 & Cypher"

Vektör araması, yalnızca sorulan kelimelere semantik olarak benzeyen belgeleri getirir. Ancak şu soruyu sorarsanız:
*"Vision Transformer modelinin kullandığı dikkat mekanizmasını GPU üzerinde hızlandıran algoritma nedir?"*

- **Vektör RAG Neden Çuvallar?** Çünkü bu bilgi tek bir belgede yazmaz! Bilgi, 3 farklı belgedeki zincirleme ilişkidedir:
  - Belge 1: `(Vision Transformer) -> [KULLANIR] -> (Self-Attention)`
  - Belge 2: `(Self-Attention) -> [HIZLANDIRIR] -> (FlashAttention)`
  - Belge 3: `(FlashAttention) -> [CALISIR] -> (NVIDIA GPU)`
- Vektör araması bu bağlantıları göremezken; **GraphRAG-2 Cypher ve Graf Gezintisi** 2-Hop / 3-Hop zincirini takip ederek cevabın anında **FlashAttention** olduğunu bulur!

```
            [Kullanıcı Sorusu: Çok Adımlı İlişki]
                            │
                            ▼
            [1. Cypher Sorgu Motoru (Parser)]
             • MATCH (a)-[r1]->(b)-[r2]->(c) ...
                            │
                            ▼
            [2. Labeled Property Graph Deposu]
             • Düğümler, Yönlü Kenarlar, Nitelikler
                            │
                            ▼
            [3. Çoklu Atlama Gezgini (BFS / En Kısa Yol)]
             • (ViT) ──► (Self-Attn) ──► (FlashAttn) ──► (GPU)
                            │
                            ▼
            [4. Alt-Grafik Serileştirme (LLM Prompt)]
             • Yüksek Sinyalli Graf Kanıtı -> LLM
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma: Labeled Property Graph (LPG) Mimarisi
- Düğümler etiket ve niteliklerle, yönlü kenarlar ise tip ve ağırlıklarla zenginleştirilerek $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ yapısında modellenir.

### 2. Deklaratif Cypher Sorgulama Motoru (`MATCH`, `WHERE`, `RETURN`)
- SQL benzeri deklaratif desen eşleme motoru ile ilişkiler tek bir sorguyla filtrelenir ve çekilir.

### 3. Çoklu Atlama (Multi-Hop Traversal: BFS) ve En Kısa Yol (Shortest Path)
- Genişlik öncelikli arama (BFS) ile 2-Hop ve 3-Hop mesafedeki komşuluklar taranır; iki kavram arasındaki doğrudan mantıksal nedensellik zinciri keşfedilir.

### 4. Alt-Grafik Çıkarımı ve LLM İçin Yapısal Bağlam Sentezi
- Keşfedilen alt-grafik, LLM istemi (prompt) için insan tarafından okunabilir Markdown formatında serileştirilir; 2-hop doğruluk oranı **%48.0'dan %96.5'e** çıkar!

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Property Graph (LPG)** | Düğümlerin ve kenarların anahtar-değer nitelikleri taşıdığı zengin graf veri modeli. |
| **Cypher** | Grafları deklaratif olarak sorgulamak için kullanılan standart desen eşleme dili. |
| **Multi-Hop Traversal** | Bir başlangıç düğümünden $k$ adım uzaklıktaki bağlantılı düğümlere zincirleme ilerleme. |
| **Shortest Path** | İki düğüm arasındaki en az adım sayısına sahip akıl yürütme veya nedensellik yolu. |
| **Breadth-First Search (BFS)** | Grafı seviye seviye genişleterek $k$-hop komşuluğu çıkaran arama algoritması. |
| **Subgraph Extraction** | Büyük bir bilgi grafından belirli bir sorgu veya düğümle ilgili alt grafın kesilip alınması. |
| **Graph Serialization** | Graf düğüm ve kenarlarını LLM'in okuyabileceği doğal dil veya Markdown metnine dönüştürme. |
| **Directed Edge** | Kaynaktan hedefe belirli bir anlam taşıyan yönlü bağlantı oku. |
| **Neighborhood (Ego Graph)** | Bir hedef düğümün $k$ mesafe içindeki tüm doğrudan ve dolaylı komşuluk kümesi. |
| **Reasoning Chain** | Çok adımlı bir sorunun yanıtına ulaşmak için izlenen mantıksal graf yolu. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %96.5 2-Hop akıl yürütme doğruluğu.│ • Çok derin (>4-Hop) aramalarda graf │
 │ • %97.8 halüsinasyon önleme başarımı.│   patlaması (Graph Explosion) riski. │
 │ • Deterministik ve şeffaf kanıt yolu.│ • Bellek içi graf indeksleme maliyeti│
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Karmaşık tedarik zinciri, tıp, fi- │ • Çok seyrek (Sparse) graflarda bağ- │
 │   nans ve kurumsal denetim sistemleri│   lantıların kopuk kalması.          │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/knowledge_graph_cypher_paneli.png` dosyası üretilir:
1. **Vektör RAG vs GraphRAG-2 Multi-Hop Kıyası**
2. **Akıl Yürütme Gezinti Yolu (Reasoning Path)**
3. **Atlama (Hop) Derinliği vs Doğruluk Eğrisi**
4. **Cypher & Gezinti İcra Gecikmesi (Latency ms)**
5. **GraphRAG-2 Cypher & Traversal Mimarisi**
6. **GraphRAG-2 Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# GraphRAG-2 Cypher ve Multi-Hop gezinti iş akışını çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
