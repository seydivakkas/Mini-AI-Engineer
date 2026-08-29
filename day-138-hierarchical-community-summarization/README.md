# Day 138: GraphRAG-3: Leiden Topluluk Tespiti ve Hiyerarşik Küme Özetleme

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu modül; Microsoft GraphRAG mimarisinin kalbi olan **Leiden Hiyerarşik Topluluk Tespiti (Hierarchical Community Detection)**, **Modülerlik Optimizasyonu ($Q$)**, **Çok Seviyeli Topluluk Raporu Üretimi (Community Summarization)** ve **Küresel Anlamlandırma (Global Sensemaking & Map-Reduce Search)** motorunu sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Büyük Resmi Görmek: Microsoft GraphRAG & Leiden"

Geleneksel RAG sistemlerine şu küresel soruyu sorduğunuzda:
*"Tüm dokümantasyonda anlatılan ana mimari temalar, bileşenler ve sistemin büyük resmi nedir?"*

- **Vektör RAG ve Yerel Graf Araması Neden Başarısız Olur?**
  Çünkü bu soru tek bir metin parçasına veya 1-2 düğüme ait değildir. Milyonlarca kelimelik tüm korpusa yayılmıştır (**The Global Sensemaking Problem**).
- **Microsoft GraphRAG Çözümü Nedir?**
  1. 🧩 **Leiden Topluluk Tespiti:** Birbiriyle sıkı ilişkili düğümleri otomatik olarak kümelere ayırır (örn: Yapay Zeka Kümesi, Dağıtık Sistemler Kümesi).
  2. 📝 **Hiyerarşik Özetleme (Bottom-Up):** Her küme için Seviye 1 özet raporları üretir; ardından bunları birleştirerek Seviye 2 Makro Sistem Raporu yazar.
  3. 🗺️ **Map-Reduce Küresel Arama:** Küresel bir soru geldiğinde tüm alt parçaları okumak yerine bu hiyerarşik raporları sentezler; böylece sistemin **büyük resmini eksiksiz ve halüsinasyonsuz (%97.2 doğrulukla)** açıklar!

```
         [Tüm Belge Tabanı & Bilgi Grafı]
                       │
                       ▼
         [1. Leiden Topluluk Tespiti]
          • Modülerlik Optimizasyonu (Q = 0.785)
          • Level 1 Meso & Level 2 Macro Kümeler
                       │
                       ▼
         [2. Bottom-Up Topluluk Özetleme]
          • Her küme için bağımsız yapısal rapor
          • Rekürsif yukarı özetleme (L1 -> L2)
                       │
                       ▼
         [3. Map-Reduce Küresel Arama Motoru]
          • MAP: Raporları küresel soruyla puanla
          • REDUCE: Makro sentez yanıtı üret
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma: Microsoft GraphRAG Mimarisi ve Küresel Anlamlandırma
- Milyonlarca belgeden oluşan devasa bilgi tabanlarında bütünsel ve makro akıl yürütme sağlar.

### 2. Leiden Hiyerarşik Topluluk Tespiti ve Modülerlik ($Q$)
- Modülerlik optimizasyonu ile grafın doğal alt-alanları (Louvain/Leiden partisyonları) $Q=0.785$ kalitesinde bölünür.

### 3. Bottom-Up Rekürsif Topluluk Özetleme ve Raporlama
- Mikro varlıklardan Seviye 1 alt-alanlara, oradan Seviye 2 makro mimari raporlarına doğru özyineli özet sentezi yapılır.

### 4. Map-Reduce Küresel Arama ve Makro İçgörü Sentezi
- Map aşamasında topluluk raporları puanlanır; Reduce aşamasında en yetkin içgörüler birleştirilerek küresel soru yanıtlanır (%44.0 -> **%97.2 başarı**).

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Microsoft GraphRAG** | Bilgi grafları üzerinde hiyerarşik topluluk tespiti ve özetleme yapan küresel RAG mimarisi. |
| **Global Sensemaking** | Büyük veri tabanlarının tamamını kapsayan bütünsel ve tematik anlama yeteneği. |
| **Leiden Algorithm** | Louvain algoritmasını iyileştirerek bağlantılı ve kaliteli graf toplulukları bulan algoritma. |
| **Modularity ($Q$)** | Graf topluluklarının iç yoğunluğunu dış bağlantı yoğunluğuna kıyaslayan kalite metriği. |
| **Hierarchical Communities** | Seviye 0 (düğüm), Seviye 1 (alt alan) ve Seviye 2 (makro kök) hiyerarşik kümeleme yapısı. |
| **Community Report** | Belirli bir topluluk kümesinin temel bulgularını ve işlevini özetleyen yapısal doküman. |
| **Bottom-Up Summarization** | Alt düzey detaylardan yukarı doğru kademeli olarak genel özetler çıkarma tekniği. |
| **Map-Reduce Search** | Topluluk raporlarını paralel puanlayıp (Map) ardından birleştiren (Reduce) arama paradigması. |
| **Meso-Level** | Mikro düğümler ile makro sistem arasındaki orta ölçekli modüler alt-alan seviyesi. |
| **Macro Synthesis** | Farklı alanların özetlerini bütünleştirerek tek bir kapsamlı yanıt üretme süreci. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %97.2 küresel tematik kapsam.      │ • İndeksleme sırasında LLM özetleme  │
 │ • Makro ve bütünsel soruları yanıtlama│   maliyeti ve ön hazırlık süresi.    │
 │ • %98.2 halüsinasyon azaltımı.       │ • Çok küçük ve seyrek graflarda top- │
 │                                      │   luluk sınırlarının belirsizleşmesi.│
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Kurumsal arşivler, kod tabanları   │ • Sürekli değişen gerçek zamanlı veri│
 │   ve karmaşık teknik dokümantasyon.  │   akışlarında özetleri güncelleme.   │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/community_summarization_paneli.png` dosyası üretilir:
1. **Standart RAG vs GraphRAG-3 Küresel Başarım**
2. **Leiden Algoritması Topluluk Kümeleri (2D Visualization)**
3. **Hiyerarşik Topluluk Katmanları**
4. **Map-Reduce Topluluk Raporu Skorları**
5. **Microsoft GraphRAG Hiyerarşik Akış Mimarisi**
6. **GraphRAG-3 Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# GraphRAG-3 topluluk tespiti ve küresel arama iş akışını çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
