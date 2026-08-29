# Day 136: GraphRAG-1: Metinden Varlık (Entity) ve İlişki (Relationship) Çıkarma

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu modül; yapılandırılmamış serbest metinleri yapısal bir Bilgi Grafına (**Knowledge Graph**) dönüştüren **GraphRAG-1: Varlık (Entity) ve İlişki (Relationship) Çıkarma Boru Hattı**, **Özne-Yüklem-Nesne (Subject-Predicate-Object) Üçlüleri (Triplets)**, **Varlık Çözümleme ve Tekilleştirme (Entity Resolution & Canonicalization)** motorunu sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Metinleri Bilgi Ağına Dönüştürmek: GraphRAG-1"

Geleneksel Vektör RAG sistemleri metinleri birbirinden kopuk "parçalar" (chunks) halinde saklar. Bu durum iki temel zaaf yaratır:
- **Çoklu Atlama (Multi-hop) Zaafı:** Eğer A belgesinde "Raft lider seçer", B belgesinde "Lider düğüm veriyi çoğaltır" yazıyorsa; standart vektör arama bu iki belge arasındaki bağlantıyı kuramaz.
- **Kavramsal Ağ Eksikliği:** Vektör araması yalnızca yerel benzerliği bulur; sistemin büyük resmini ve tüm kavramların birbirine nasıl bağlandığını bilemez.

**GraphRAG-1 Nasıl Çalışır?**
1. 🏷️ **Varlık Çıkarımı (Entities / Nodes):** Metindeki kritik kavramlar, algoritmalar ve teknolojiler (`Raft`, `Vision Transformer`, `PostgreSQL`) tespit edilir.
2. 🔗 **İlişki Çıkarımı (Triplets / Edges):** Bu varlıklar arasındaki eylemler yönlü üçlüler halinde çıkarılır:  
   `(Vision Transformer) ──[KULLANIR]──► (Self-Attention)`
3. 🔄 **Varlık Çözümleme (Entity Resolution):** Eşanlamlılar (`ViT`, `Vision Transformers`) tek bir kanonik düğümde (`Vision Transformer`) birleştirilir.
4. 🌐 **Bilgi Grafı (Knowledge Graph):** Metin, sorgulanabilir ve analiz edilebilir dinamik bir graf yapısına $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ dönüştürülür!

```
            [Yapılandırılmamış Metin]
                       │
                       ▼
        [1. Varlık Çıkarıcı (Entity Extractor)]
         • Düğümler (Nodes): Algoritma, Teknoloji
                       │
                       ▼
        [2. İlişki Çıkarıcı (Triplet Extractor)]
         • Kenarlar: (Özne, YÜKLEM, Nesne, Ağırlık)
                       │
                       ▼
        [3. Varlık Çözümleme (Entity Resolution)]
         • ViT -> Vision Transformer (Tekilleştirme)
                       │
                       ▼
        [4. Yapısal Bilgi Grafı (Knowledge Graph)]
         G = (V, E) -> Çoklu Atlama ve Sorgulama
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma: Vektör RAG Kısıtları ve GraphRAG Devrimi
- Graf yapıları $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, belgeler arasındaki gizli ve dolaylı ilişkileri zincirleme bağlar.

### 2. Yapısal Varlık Çıkarımı ve Tip Sınıflandırması
- Varlıklar `ALGORITMA`, `TEKNOLOJI`, `KAVRAM` ve `METRIK` kategorilerine ayrılarak tip ve açıklamalarıyla zenginleştirilir.

### 3. Yönlü İlişki Üçlüleri (Directed Triplets: $(s, r, o, w)$)
- `(Özne, YÜKLEM, Nesne)` formatında yönlü ve ağırlıklı kenarlar kurularak semantik doğruluk garanti edilir.

### 4. Varlık Çözümleme ve Tekilleştirme (Entity Resolution)
- Eşanlamlılar ve kısaltmalar kanonik tek bir düğüme indirgenir; F1 çıkarma başarımı **%62.0'dan %96.8'e** fırlar.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **GraphRAG** | Vektör aramasını Bilgi Grafları (Knowledge Graph) ile zenginleştiren yeni nesil RAG mimarisi. |
| **Knowledge Graph ($\mathcal{G}$)** | Varlıkların düğüm ($\mathcal{V}$), ilişkilerin kenar ($\mathcal{E}$) olarak modellendiği yapısal graf. |
| **Entity (Varlık)** | Metindeki önemli kişi, teknoloji, algoritma veya kavram düğümü. |
| **Triplet (Üçlü)** | `(Subject, Predicate, Object)` formatındaki yönlü semantik ilişki ifadesi. |
| **Entity Resolution** | Farklı isimlerle geçen aynı varlığı tek bir kanonik düğümde birleştirme süreci. |
| **Multi-hop Reasoning** | Graf üzerinde birden fazla kenarı takip ederek dolaylı soruları yanıtlama yeteneği. |
| **Degree Centrality** | Bir düğümün diğer düğümlerle kurduğu toplam bağlantı sayısı / merkezilik derecesi. |
| **Canonical Name** | Bir varlığın sistem genelinde kabul edilen standart resmi adı. |
| **Predicate (Yüklem)** | İki varlık arasındaki ilişkinin türü (`KULLANIR`, `ENGELLER`, `HIZLANDIRIR`). |
| **Edge Weight** | Bir ilişkinin metin içindeki geçiş sıklığına veya güven katsayısına dayalı ağırlığı. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %96.8 yüksek varlık F1 başarımı.   │ • Ham vektör aramaya göre ön işlem   │
 │ • Çoklu atlama (Multi-hop) yeteneği. │   (İndeksleme) süresinin daha uzun   │
 │ • Belgeler arası küresel bağlantılar.│   olması.                            │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Kurumsal dokümantasyon, tıp ve fi- │ • Çok karmaşık cümlelerde örtük iliş-│
 │   nans alanlarında karmaşık sorular. │   kilerin eksik çıkarılma ihtimali.  │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/graph_rag_entity_extraction_paneli.png` dosyası üretilir:
1. **Standart NER vs GraphRAG-1 Başarımı**
2. **Çıkarılan Bilgi Grafı (2D Knowledge Graph Network)**
3. **Varlık Tipleri Dağılımı (Entity Categories)**
4. **Varlık Çözümleme (Entity Canonicalization)**
5. **GraphRAG-1 Çıkarım Hattı Mimarisi**
6. **GraphRAG-1 Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# GraphRAG-1 varlık çıkarma iş akışını çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
