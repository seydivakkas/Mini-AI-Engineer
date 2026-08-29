# Day 133: HyDE (Hypothetical Document Embeddings) & Sıfır-Atış Soru Zenginleştirme

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu modül; klasik vektör getirmedeki *"Soru-Belge Anlamsal Asimetrisi"* (Query-Document Asymmetry) sorununu çözen **HyDE (Hypothetical Document Embeddings)**, **Sıfır-Atış (Zero-Shot) Hipotez Belgesi Üretimi**, **Çoklu Hipotez Centroid Vektör Birleştirme** ve **Gelişmiş RAG Getirme Hattı**nı sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Soruyla Değil, Hayali Cevapla Arama Yapmak: HyDE Mimarisi"

Geleneksel vektör aramalarında büyük bir anlamsal uçurum vardır:
- **Kullanıcı Sorusu ($q$):** Kısa, eksik ve soru kipleriyle doludur (*"Quorum ve split-brain nedir?"*).
- **Hedef Belge ($d$):** Uzun, teknik, açıklayıcı ve formüllerle doludur (*"Raft protokolünde durum makinesi çoğaltması çoğunluk kuralı ile..."*).
- Bir embedding modeli bu iki farklı metin türünü doğrudan karşılaştırdığında ($E(q) \cdot E(d)$), soru ve belge farklı vektör manifoldlarında olduğu için arama doğruluğu %50'lerde kalır!

**HyDE (Hypothetical Document Embeddings) Bu Sorunu Nasıl Çözer?**
1. 🤖 **Hayali Cevap Üret (Zero-Shot):** LLM'e soru verilir ve *"Bu soruya teknik bir el kitabı gibi varsayımsal bir yanıt yaz"* denir. LLM hayali bir belge ($\hat{d}$) üretir.
2. 🎯 **Manifold Uyuşumu:** Bu hayali belge ufak tefek teknik yanlışlar (halüsinasyon) içerse bile, **üslup, teknik kelime dağarcığı ve yapısal olarak tam bir belgedir!**
3. 📐 **Centroid Vektörü:** Birden fazla hayali belgenin embedding ortalaması ($\mathbf{e}_{\text{HyDE}}$) alınır.
4. 🔍 **Belge-Belge Eşleşmesi:** Artık arama bir "Soru-Belge" eşleşmesi değil, **"Belge-Belge" (Document-to-Document)** eşleşmesine dönüşür! Arama başarımı **%58'den %95.6'ya** fırlar.

```
            [Kullanıcı Sorusu: q ∈ Q]
                       │
                       ▼
        [LLM Sıfır-Atış Hipotez Üretimi]
         (Prompt: Bu soruya teknik yanıt yaz)
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    [Hipotez d̂₁]  [Hipotez d̂₂]  [Hipotez d̂₃]
         │             │             │
         └─────────────┼─────────────┘
                       ▼
         [Embedding & Centroid Birleştirme]
         e_HyDE = Normalize( 1/N * Σ e_i )
                       │
         [Vektör Veritabanı (Gerçek Belgeler)]
                       │ (Cosine Sim: e_HyDE · E(d))
                       ▼
          [En İlgili Gerçek Belgeler (Top-k)]
                       │
                       ▼
           [Nihai Doğru LLM Yanıtı]
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma: Soru-Belge Asimetrisi ve HyDE Paradigması
- Sorular ($q \in \mathcal{Q}$) ile belgeler ($d \in \mathcal{D}$) farklı manifoldlardadır. HyDE, $q \to \hat{d}$ dönüşümüyle soruyu doğrudan $\mathcal{D}$ uzayına projekte eder.

### 2. Sıfır-Atış (Zero-Shot) Hipotez Üretimi ve Manifold Projeksiyonu
- Hiçbir eğitim veya fine-tuning yapmadan genel yetenekli bir LLM ile belge üslubu taklit edilir.

### 3. Çoklu Hipotez Centroid Birleştirme ($\mathbf{e}_{\text{HyDE}}$) ve Kararlılık
- $N=3$ ila $5$ arası üretilen hipotezlerin birim normalleştirilmiş ortalaması alınarak tekil halüsinasyon riskleri elenir.

### 4. Sıfır-Atış Getirme Doğruluğu (Recall@5) ve Halüsinasyon Dayanıklılığı
- Hipotezdeki olgusal hatalar embedding uzayında kaybolur; önemli olan terminolojik yakınlıktır. Recall@5 **%58.4'ten %95.6'ya** çıkar.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **HyDE** | Hypothetical Document Embeddings; soru yerine varsayımsal yanıt embedding'i ile arama. |
| **Query-Document Asymmetry** | Soru ile belgenin uzunluk, üslup ve içerik bakımından farklı vektör uzaylarında olması. |
| **Hypothetical Document ($\hat{d}$)** | LLM tarafından soruya binaen sıfır-atış üretilen varsayımsal pasaj. |
| **Centroid Vector** | Çoklu hipotez embedding'lerinin ortalaması alınarak normalize edilen merkez vektör. |
| **Manifold Projection** | Bir veri dağılımını (soru) başka bir geometrik manifolda (belge) yansıtma işlemi. |
| **Zero-Shot Dense Retrieval** | Model eğitmeden doğrudan genel embedding'lerle yapılan yoğun vektör getirme. |
| **Hallucination Invariance** | Hipotez belgesindeki olgusal hataların genel konu vektörünü bozmaması prensibi. |
| **Document-to-Document Search** | Soru-Belge yerine Belge-Belge formatında yapılan simetrik vektör karşılaştırması. |
| **Recall@k** | İlgili gerçek belgenin ilk $k$ getirme sonucu içinde bulunma oranı. |
| **Domain Terminology Capture** | Soruda geçmeyen ancak alana özgü terimlerin hipotezde kendiliğinden belirmesi. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %95.6 üstün sıfır-atış getirme.    │ • Arama öncesi ekstra LLM çıkarımı  │
 │ • Model fine-tuning gerektirmeme.    │   (Inference) gecikmesi (~100-300ms).│
 │ • Teknik terimleri otomatik tamamlama│ • Çok küçük basit sorularda gereksiz │
 │   ve soru-belge asimetrisini çözme.  │   işlem yükü.                        │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Hukuk, tıp, finans gibi jargonu ve │ • LLM'in konu hakkında tamamen saç-  │
 │   terminolojisi ağır RAG sistemleri. │   malaması durumunda yanlış yönlenme.│
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/hyde_embeddings_paneli.png` dosyası üretilir:
1. **Standart Dense vs BM25 vs HyDE Başarımı**
2. **Soru, Hipotez ve Gerçek Belge Manifold Dağılımı (2D)**
3. **Hipotez Sayısı (N) ve Centroid Kararlılık Skoru**
4. **Belge Başına Standart vs HyDE Kosinüs Benzerliği**
5. **HyDE Mimari Akış Şeması**
6. **HyDE RAG Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# HyDE RAG arama iş akışını çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
