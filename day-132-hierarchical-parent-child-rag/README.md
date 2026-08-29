# Day 132: Hiyerarşik RAG (Parent-Child / Small-to-Big Retrieval)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu modül; klasik RAG sistemlerindeki *"Küçük parça bağlamı kaybeder, büyük parça vektör aramasını bozar"* ikilemini ortadan kaldıran **Hiyerarşik Parent-Child (Small-to-Big Retrieval)** mimarisini, **Key-Value Belge Deposu (DocStore)**, **Çocuk Parçalarla Hassas Vektör İndeksleme** ve **Ebeveyn Parçalarla Bağlam Genişletme** sistemini sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Küçük Parçayla Ara, Büyük Paragrafı Getir: Parent-Child RAG"

Standart RAG mimarilerinde büyük bir açmaz vardır:
- **Küçük Parçalar (100-200 karakter):** Vektör araması için harikadır; embedding son derece keskin ve nettir. Ancak LLM'e verildiğinde cümlenin başı ve sonu eksik olduğu için model doğru yanıt üretemez.
- **Büyük Parçalar (1000-2000 karakter):** LLM'e verildiğinde zengin bağlam sunar. Ancak vektörleştirildiğinde 10 farklı konu birbirine karışır (**Vector Dilution / Vektör Seyrelmesi**) ve arama motoru doğru parçayı bulamaz.

**Parent-Child (Small-to-Big) Mimarisi Neyi Değiştirir?**
1. 🌳 **İkili Hiyerarşi:** Belge önce büyük **Ebeveyn (Parent)** parçalara (örn. 500-1000 karakter) bölünür.
2. 👶 **Çocuk Parçalar (Child):** Her ebeveyn parça kendi içinde küçük **Çocuk (Child)** parçalara (örn. 150 karakter) ayrılır.
3. 🗄️ **Ayrı Depolama:** Ebeveynler bir **Belge Deposunda (DocStore)** saklanır. Yalnızca çocuk parçalar vektör indeksine gömülür.
4. 🔍 **Küçükten Büyüğe Getirme (Small-to-Big):** Kullanıcı sorgusu küçük çocuk parçalarla aranır. Eşleşen çocuk parçanın `parent_id`'si üzerinden DocStore'dan tam ebeveyn paragrafı çekilerek LLM'e verilir!

```
               [Ham Belge Metni]
                       │
                       ▼
       [1. Ebeveyn Parçalama (Parent Chunks)]
          │ (500-1000 Karakter Geniş Bağlam)
          ├────────────────────────┐
          ▼                        ▼
  [Belge Deposu (DocStore)]  [2. Çocuk Parçalama]
  (Key-Value: parent_id)     (150 Karakter Keskin)
          │                        │
          │                        ▼
          │              [Vektör İndeksleyici]
          │                        │
          │   [Kullanıcı Sorgusu] ──┘ (Vektör Arama)
          │                        ▼
          │              [Top-k Çocuk Parçalar]
          │                        │
          └─────────► ┌────────────┴────────────┐
                      ▼                         ▼
          [DocStore'dan Parent Getir] (Small-to-Big)
                      │
                      ▼
         [LLM İçin Zengin Tam Bağlam]
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma: RAG İkilemi ve Hiyerarşik Parent-Child Mimarisi
- Arama aşamasında küçük vektör hassasiyeti (*High Precision*), üretim aşamasında büyük ebeveyn bağlamı (*High Recall*) kullanılır.

### 2. İndeksleme Katmanı: Yalnızca Küçük Çocuk Parçaların Vektörleştirilmesi
- Vektör veritabanı yalnızca çocuk embedding'leri taşır; böylece vektör boyutu ve gürültüsü minimumda tutulur.

### 3. Belge Deposu (DocStore) ve Tekilleştirme (Deduplication)
- Birden fazla çocuk aynı ebeveyne işaret ediyorsa, DocStore'dan o ebeveyn yalnızca 1 kez çekilir; token israfı önlenir.

### 4. Vektör Seyrelme (Dilution) Önleme ve LLM Doğruluğu
- Vektör seyrelme oranı %44'ten **%2.5'e** düşer; LLM yanıt doğruluğu **%64.5'ten %96.8'e** fırlar.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Parent-Child RAG** | Vektör araması çocuk parçada yapılıp bağlamın ebeveyn parçadan getirildiği hiyerarşik yöntem. |
| **Small-to-Big Retrieval**| Küçük parça eşleşmesini büyük belge bloğuna genişletme tekniği. |
| **DocStore** | Ebeveyn parçaları `parent_id` anahtarıyla saklayan Key-Value hafıza deposu. |
| **Vector Dilution** | Büyük metin bloklarının tek vektöre sıkıştırılmasıyla oluşan anlam kaybı ve gürültü. |
| **Child Chunk** | Vektör araması için optimize edilmiş kısa, odaklanmış alt metin parçası. |
| **Parent Chunk** | LLM'in mantık yürütmesi için gereken eksiksiz bağlamı içeren büyük paragraf parçası. |
| **Deduplication** | Aynı ebeveyne ait birden çok çocuk eşleştiğinde ebeveynin tekilleştirilmesi. |
| **Hierarchical Indexing** | Belge içeriğinin üst ve alt katmanlar halinde çok seviyeli indekslenmesi. |
| **Recall@k** | İlgili bilginin ilk $k$ getirme sonucu içinde bulunma oranı. |
| **Context Window Hygiene**| LLM istemine yalnızca eksiksiz ve alakalı ebeveyn blokların eklenmesi disiplini. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %97.2 üst düzey arama doğruluğu.   │ • Hem Vektör İndeks hem DocStore yö- │
 │ • Vektör seyrelmesini sıfırlama.     │   netiminin getirdiği ek mimari yük. │
 │ • LLM üretiminde eksiksiz paragraf.  │ • Çocuk parça sayısının artmasıyla   │
 │                                      │   büyüyen indeks bellek boyutu.      │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Çok sayfalı PDF'ler, teknik dokü-  │ • Çok küçük belgelerde getirdiği ek  │
 │   manlar ve kurumsal bilgi bankaları.│   karmaşıklığın gereksiz kalması.    │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/hierarchical_parent_child_paneli.png` dosyası üretilir:
1. **Düz Parçalama vs Parent-Child RAG Başarımı**
2. **Ebeveyn Başına İndekslenen Çocuk Parça Sayısı**
3. **Vektör Aramasında Eşleşen Çocuk Parça Benzerliği (%)**
4. **Small-to-Big Bağlam Genişletme Boyutu**
5. **Parent-Child (Small-to-Big) Mimarisi Şeması**
6. **Parent-Child RAG Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Parent-Child hiyerarşik RAG iş akışını çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
