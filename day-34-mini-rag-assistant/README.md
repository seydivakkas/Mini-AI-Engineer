# Day 34: Mini RAG Asistanı & Doküman Soru-Cevap Motoru (Mini RAG Assistant)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; kurumsal teknik dokümanlar, fabrika bakım kılavuzları ve veri tabanları üzerinde **üretken yapay zeka modellerinin halüsinasyon görmesini engelleyen**, kaynak atıflı (citation-backed) ve güvenli yanıt üreten **Mini RAG (Retrieval-Augmented Generation)** motorunun sıfırdan geliştirilmiş uçtan uca mimarisidir.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Neden RAG (Retrieval-Augmented Generation)?
Büyük Dil Modelleri (LLM'ler), eğitim verileriyle sınırlıdır; kurum içi özel dokümanları bilmezler ve emin olmadıkları durumlarda gerçeğe aykırı ancak ikna edici **halüsinasyonlar (hallucinations)** üretirler.
RAG mimarisi şu 3 temel aşamayla bu sorunu çözer:
1. **İndeksleme & Parçalama (Indexing & Chunking):** Uzun metinlerin anlamsal bütünlüğü korunarak çakışmalı (sliding window) parçalara bölünmesi.
2. **Anlamsal Bilgi Erişimi (Retrieval):** Kullanıcı sorusu ile doküman parçaları arasındaki kosinüs benzerliğini hesaplayarak en alakalı Top-$K$ parçanın getirilmesi.
3. **Bağlam Enjeksiyonu & Sentez (Context Injection & Synthesis):** Seçilen parçaların sistem prompt'una enjekte edilip kaynak atıflı (`[Kaynak: CHUNK_ID]`) yanıt üretilmesi.

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                 KULLANICI SORUSU (QUERY)                 │
                    └────────────────────────────┬─────────────────────────────┘
                                                 │
                                                 ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │  1. VEKTÖR BİLGİ ERİŞİMİ (RETRIEVAL)                                         │
        │  - Soru Embedding Vektörünün Çıkarılması                                     │
        │  - Vektör Deposunda Kosinüs Benzerliği Araması                               │
        │  - Top-K En İlgili Doküman Parçasının Seçilmesi                              │
        └────────────────────────────────────────┬─────────────────────────────────────┘
                                                 │ (Top-K İlgili Parçalar)
                                                 ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │  2. BAĞLAM ENJEKSİYONU (CONTEXT INJECTION & PROMPT ASSEMBLY)                 │
        │  - Sistem Talimatı: "Yalnızca verilen bağlamdaki gerçeklere dayan."          │
        │  - Bağlam Blokları: [Kaynak: KB-001_chunk_00] Kılavuz metni...               │
        │  - Kullanıcı Sorusu Enjeksiyonu                                              │
        └────────────────────────────────────────┬─────────────────────────────────────┘
                                                 │
                                                 ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │  3. KAYNAK ATIFLI YANIT SENTEZİ & HALÜSİNASYON FİLTRESİ                      │
        │  - Güven Eşiği Kontrolü (Confidence Thresholding >= 0.20)                    │
        │  - Atıf Etiketleme: "Parça bilgisi... [Kaynak: KB-001_chunk_00]"             │
        │  - Bilgi Yetersizse Güvenli Ret (Refusal)                                    │
        └────────────────────────────────────────┬─────────────────────────────────────┘
                                                 │
                                                 ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │  DOĞRULANABİLİR, HALÜSİNASYONSUZ UZMAN YANITI            │
                    └──────────────────────────────────────────────────────────┘
```

---

### 2. Metin Parçalama (Chunking) ve Kayan Pencere (Sliding Window)

Uzun metinleri tek parça halinde vektörleştirmek anlamsal detayı boğar. Bu nedenle metinler `chunk_size` ve `chunk_overlap` parametreleriyle parçalanır.

- **Adım Mesafesi (Stride):** $\text{Adım} = \text{Chunk\_Boyutu} - \text{Çakışma\_Miktarı}$
- **Çakışma (Overlap) Neden Hayatidir?** Cümle sınırları veya önemli teknik tanımlar tam parça bitişine denk geldiğinde cümlenin yarısı kaybolabilir. Çakışma payı, bilginin iki parça arasında güvenle taşınmasını sağlar.

```
Doküman Kelimeleri: [ w0  w1  w2  w3  w4  w5  w6  w7  w8  w9  w10 w11 w12 w13 w14 ]
Chunk 0 (0-9)     : [───────────────────────────]
Chunk 1 (7-16)    :                    [░░░░░░░░───────────────────────────]
                                       (Çakışma Payı = 3 Kelime)
```

---

### 3. Soru-Cevap & Doğrulama Deney Sonuçları

| Test Sorusu | Erişilen Parça ID | Benzerlik Skoru | Yanıt Durumu | Kaynak Atfı |
|---|---|---|---|---|
| **"RAG mimarisi nasıl çalışır?"** | `KB-002_chunk_00` | **0.4820** | `BASARILI` | `KB-002_chunk_00` |
| **"Bantlarda nesne takibi nasıl yapılır?"** | `KB-001_chunk_01` | **0.4410** | `BASARILI` | `KB-001_chunk_01` |
| **"Kuantum süperiletken kubit?"** *(Alakasız)* | `KB-003_chunk_00` | **0.0815** *(< 0.20)* | `YETERSIZ_KANIT` | *Yok (Güvenli Ret)* |

---

## 🛠️ Dizin Yapısı

```
day-34-mini-rag-assistant/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # torch, numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca RAG çalıştırma ve test betiği
├── README.md                        # Detaylı teorik ve mimari dokümantasyon (220+ Satır)
├── src/
│   ├── __init__.py
│   ├── metin_parcalayici.py         # Kayan pencereli parça bölücü (Chunker)
│   ├── vektor_deposu.py             # Vektör veritabanı ve kosinüs arama motoru
│   ├── rag_ureteci.py               # Prompt montajı, bağlam enjeksiyonu ve sentez
│   ├── rag_asistani.py              # Uçtan uca RAG yöneticisi (Orchestrator)
│   └── gorsellestirici.py           # 6 panelli RAG analiz panosu (Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_mini_rag.py             # 7 adet kapsamlı birim test
└── ciktilar/
    └── rag_analiz_paneli.png        # 6 panelli RAG analiz görseli
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

**Görev:** `src/rag_asistani.py` içerisine **"Çoklu Turlu Konuşma Belleği (Multi-Turn Chat History)"** ekleyerek kullanıcının önceki sorularını ve sistem yanıtlarını hafızada tutan ve bir sonraki soruya bağlam olarak aktaran `SohbetliRAGAsistani` sınıfını geliştirmek.

**Tamamlanan Çözüm:**
```python
class SohbetliRAGAsistani(MiniRAGAsistani):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gecmis = []

    def sohbet_et(self, soru: str, top_k: int = 2) -> dict:
        # Önceki soru-cevap geçmişini sorgu bağlamına ekle
        zenginlestirilmis_soru = f"{' '.join([g['soru'] for g in self.gecmis[-2:]])} {soru}".strip()
        cikis = self.soru_sor(zenginlestirilmis_soru, top_k=top_k)
        self.gecmis.append({"soru": soru, "yanit": cikis["yanit"]})
        return cikis
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** RAG mimarisinde parça boyutunu (`chunk_size`) çok küçük (ör. 5 kelime) veya çok büyük (ör. 2000 kelime) seçmenin ne gibi dezavantajları vardır?

> **Mentor Cevabı:**
> 1. **Aşırı Küçük Parça Boyutu ($< 15$ kelime):** Parça içerisindeki anlamsal bağlam (context) kaybolur. Model, kelimelerin ait olduğu cümlenin öznesini veya amacını anlayamaz ve bilgi erişiminde alakasız eşleşmeler yapar.
> 2. **Aşırı Büyük Parça Boyutu ($> 1000$ kelime):** Vektör uzayında embedding vektörü metnin ortalama anlamını temsil ettiği için özel detaylar kaybolur ("Lost in the Middle" problemi). Ayrıca prompt'a aşırı gürültülü metin eklendiği için modelin odaklanma ve doğru cevabı ayıklama başarısı düşer.
> 3. **İdeal Aralık:** Çoğu kurumsal doküman için 100-300 kelime (yaklaşık 256-512 token) ve %15-25 çakışma (overlap) optimum başarıyı sunar.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
