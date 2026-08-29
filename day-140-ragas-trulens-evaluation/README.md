# Day 140: Ragas & TruLens ile RAG Değerlendirmesi (RAG Triad) ve FAZ 7 BÜYÜK FİNALİ

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%207-BÜYÜK%20FİNAL%20(%25100)-gold.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; üretim seviyesindeki RAG ve Otonom Ajan sistemlerinin kalitesini matematiksel ve deterministik metriklerle ölçen **Ragas & TruLens RAG Triad Değerlendirme Çerçevesi**ni (Faithfulness, Answer Relevance, Context Precision, Context Recall), **Otomatik Karşılaştırmalı Benchmark Boru Hattı**nı ve **FAZ 7 (Otonom AI Ajanları ve Advanced GraphRAG) Büyük Finali**ni içermektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ RAG Sistemleri Neden Sezgisel Değil Matematiksel Ölçülmelidir?
Geleneksel yazılımlarda birim testler "Doğru/Yanlış" mantığıyla çalışırken, LLM tabanlı RAG sistemlerinde yanıtlar serbest metin formatındadır. Bir RAG sisteminin güvenilirliğini ölçmek için **TruLens RAG Triad** üçgeni ve **Ragas Metrikleri** kullanılır:

1. **Sadakat (Faithfulness / Groundedness):** Model uyduruyor mu (halüsinasyon), yoksa söyledikleri bağlamda geçiyor mu?
   $$\text{Faithfulness} = \frac{|\text{Desteklenen İddialar}|}{|\text{Toplam İddialar}|}$$
2. **Soru Uygunluğu (Answer Relevance):** Model kullanıcının sorduğu soruya mı cevap veriyor, yoksa konudan saptı mı?
3. **Bağlam Kapsama (Context Recall):** Arama motoru, doğru cevabı vermek için gereken tüm referans bilgileri getirdi mi?
4. **Bağlam Hassasiyeti (Context Precision@K):** Alakalı bilgi parçaları en üst sıralarda mı getirildi?

```
               [Kullanıcı Sorusu: Query]
                        │      │
      (Answer Relevance)│      │(Context Precision)
                        ▼      ▼
[Üretilen Yanıt] ◄─── [Getirilen Bağlam: Context] ◄─── [Ground Truth]
        │                      │                             │
        └────── (Faithfulness) ┴────── (Context Recall) ─────┘
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Çekirdek Mekanizma: TruLens RAG Triad Üçgeni
- TruLens mimarisi; Soru, Bağlam ve Yanıt üçlüsü arasındaki köprüleri bağımsız olarak denetler. Yanıtın bağlama sadakati ölçülürken dış dünya bilgisi yok sayılır.

### B. Atomik İddia Ayrıştırması ve Halüsinasyon Tespiti
- Üretilen yanıt bağımsız olgusal iddialara (Atomic Claims) ayrılır. Her iddia getirilen bağlam parçalarıyla çapraz doğrulamaya tabi tutulur. Desteklenmeyen iddialar **Halüsinasyon** olarak etiketlenir.

### C. Context Recall ve Context Precision@K Formülasyonu
- Context Precision, doğru parçaların sıralamadaki pozisyonunu $Precision@k$ üzerinden ağırlıklandırarak en üst sıralardaki alaka düzeyini ödüllendirir:
  $$\text{Context Precision@K} = \frac{\sum_{k=1}^K (\text{Precision@k} \times v_k)}{\text{Toplam Alakalı Parça}}$$

### D. Faz 7 Mimarilerinin Karşılaştırmalı Kıyaslaması (Comparative Benchmark)
- Faz 7 boyunca geliştirilen 4 mimarinin RAG Triad skorları:
  - **Naive Chunk RAG:** Sadakat %62.5, Uygunluk %64.0, RAG Triad %61.5 (Halüsinasyon: %37.5).
  - **Semantic Chunking + HyDE:** Sadakat %82.0, Uygunluk %86.5, RAG Triad %84.1 (Halüsinasyon: %18.0).
  - **Contextual Compression + Re-ranker:** Sadakat %93.5, Uygunluk %94.0, RAG Triad %93.0 (Halüsinasyon: %6.5).
  - **Advanced Hybrid GraphRAG (Final):** Sadakat **%98.2**, Uygunluk **%97.5**, RAG Triad **%97.5** (Halüsinasyon: **%1.8**).

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **RAG Triad** | Faithfulness, Answer Relevance ve Context Recall bileşenlerinden oluşan üçlü değerlendirme çerçevesi. |
| **Faithfulness (Sadakat)** | Üretilen yanıttaki olgusal iddiaların getirilen bağlam tarafından desteklenme oranı. |
| **Groundedness** | Yanıtın dış spekülasyon yerine sadece sağlanan bağlama dayanması durumu. |
| **Answer Relevance** | Modelin ürettiği cevabın kullanıcının asıl sorusuna anlamsal odaklanma derecesi. |
| **Context Recall** | Referans doğrudan cevabın ne kadarının getirilen bağlamda kapsandığının oranı. |
| **Context Precision@K** | Getirilen bağlam parçalarından alakalı olanların sıralamadaki öncelik kalitesi. |
| **Atomic Claim** | Doğrulanabilir tek bir olguyu ifade eden en küçük cümle veya iddia parçası. |
| **Harmonic RAG Triad Score** | Üç metriğin harmonik ortalamasıyla hesaplanan dengeli genel başarı puanı. |
| **Hallucination Rate** | Yanıtta yer alan ve bağlamda kanıtı bulunmayan uydurma iddiaların yüzdesi. |
| **Comparative Benchmark** | Farklı RAG boru hatlarını aynı altın standart veri kümesi üzerinde kıyaslama süreci. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %98.2 sadakat ve %1.8 halüsinasyon │ • Ground truth referans veri kümesi  │
 │   ile endüstriyel güvenilirlik.      │   hazırlamanın etiketleme maliyeti.  │
 │ • Tamamen otomatik ve matematiksel   │ • Çok dilli metinlerde kök eşleme    │
 │   değerlendirme boru hattı.          │   hassasiyet ihtiyacı.               │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Üretim RAG sistemlerinin CI/CD test│ • Dinamik değişen bilgi tabanlarında │
 │   süreçlerine entegre edilmesi.      │   referans doğruların eskimesi.      │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 🏆 5. FAZ 7 BÜYÜK FİNALİ RETROSPEKTİFİ (GÜN 121 - 140)

20 gün boyunca sıfırdan inşa edilen devasa Otonom Ajan & Advanced GraphRAG yol haritası:

1. **Gün 121-125:** ReAct Mantığı, Araç Entegrasyonu, Hafıza ve Reflexion Öz-Eleştiri Döngüleri.
2. **Gün 126-130:** Multi-Agent İşbirliği, Hiyerarşik Yönetici Ajanlar ve Consensus Algoritmaları.
3. **Gün 131-135:** Semantic Chunking, Parent-Child İndeksleme, HyDE, Two-Stage Cross-Encoder ve Contextual Compression.
4. **Gün 136-139:** GraphRAG Varlık/İlişki Çıkarma, Neo4j/Cypher, Leiden Hiyerarşik Topluluk Tespiti ve Hibrit RRF Füzyonu.
5. **Gün 140:** Ragas & TruLens Değerlendirme Çerçevesi ve **FAZ 7 BÜYÜK FİNALİ!**

---

## 📊 6. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/ragas_trulens_evaluation_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
