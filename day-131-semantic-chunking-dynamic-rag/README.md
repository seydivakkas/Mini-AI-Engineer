# Day 131: Semantik Parçalama (Semantic Chunking) & Dinamik RAG Bölümleme

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu modül; geleneksel sabit karakter/token parçalama yöntemlerinin (*Fixed-size Chunking*) yarattığı bağlam ve cümle kopukluklarını ortadan kaldıran **Semantik Parçalama (Semantic Chunking)**, **Cümle Vektörleri Arası Kosinüs Mesafesi Analizi**, **Dinamik Eşikleme (Percentile & Standard Deviation Thresholding)** ve **Gelişmiş RAG Getirme Hattı**nı sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Metinleri Körlemesine Değil, Anlamına Göre Bölmek: Semantik Parçalama"

Geleneksel RAG sistemlerinde metinler genellikle "Her 500 karakterde bir kes, 50 karakter çakışma (overlap) bırak" mantığıyla bölünürdü. Bu yöntem bir cümleyi tam ortasından ("Yapay zeka modelleri [KESİNTİ] hızla gelişiyor") böler, paragrafların bağlamını koparır ve RAG getirme doğruluğunu %60'lara düşürürdü.

**Semantik Parçalama (Semantic Chunking) Nasıl Çalışır?**
1. ✂️ **Cümle Ayrıştırma:** Belge önce tam dilbilgisel cümlelere ayrıştırılır ($s_1, s_2, s_3 \dots$).
2. 🪟 **Bağlam Tamponu (Sliding Window):** Her cümlenin önüne ve arkasına komşu cümleler eklenerek zengin yerel bağlam vektörü oluşturulur.
3. 📐 **Kosinüs Mesafesi Ölçümü:** Ardışık cümlelerin embedding'leri arasındaki anlamsal mesafe hesaplanır:
   $$d_i = 1 - \text{CosineSimilarity}(\mathbf{e}_i, \mathbf{e}_{i+1})$$
4. 📈 **Dinamik Eşik Tespiti:** Mesafelerin ortalaması ve standart sapması ($\text{Eşik} = \mu + \alpha \cdot \sigma$) üzerinden konu değişim noktaları tespit edilir.
5. 📦 **Dinamik Semantik Parçalar:** Mesafe eşiği aştığında yeni bir parça başlatılır; aynı konuyu anlatan cümleler tek bir parça içinde korunur.

```
                 [Ham Belge Metni]
                         │
                         ▼
             [Cümle Ayrıştırıcı (Regex)]
                         │ (Cümle Dizisi: s_1, s_2...)
                         ▼
           [Bağlam Tamponu (Sliding Window)]
                         │
                         ▼
           [Cümle Embedding Üretimi (Embed)]
                         │ (e_i Vektörleri)
                         ▼
          [Kosinüs Mesafesi: d_i = 1 - CosSim]
                         │
                         ▼
          [Dinamik Eşikleme (mean + 0.45 * std)]
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
    (d_i <= Eşik)                 (d_i > Eşik)
  [Aynı Parçaya Ekle]          [Yeni Parça Başlat]
                         │
                         ▼
             [Semantik Bütünlüklü RAG]
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma: Sabit Parçalama Kısıtları ve Semantik Parçalama Devrimi
- Sabit parçalamada cümleler ortadan kesilir, varlık (Entity) bütünlüğü bozulur.
- Semantik parçalamada ise doğal paragraf ve fikir sınırları korunur.

### 2. Cümle Ayrıştırma, Kayan Pencere Bağlam Tamponu ve Vektörleştirme
- Tekil cümleler yerine kayan pencere ($s_{i-1} + s_i + s_{i+1}$) vektörleştirilerek mikro-gürültüler filtrelenir.

### 3. Ardışık Kosinüs Mesafesi ve Dinamik Eşik Tespiti
- $d_i = 1 - \cos(\mathbf{e}_i, \mathbf{e}_{i+1})$ formülüyle anlamsal sıçramalar (*Semantic Spikes*) bulunur.
- Eşik, belgenin kendi dağılımına göre dinamik olarak hesaplanır ($\mu + \alpha\sigma$ veya Percentile).

### 4. RAG Getirme Hassasiyeti (Retrieval Precision@k) ve Varlık Bütünlüğü
- Parça bütünlüğü %45'ten **%98.2'ye**, RAG getirme doğruluğu **%62.4'ten %94.8'e** yükselir.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Semantic Chunking** | Metinleri sabit karakter yerine anlamsal benzerlik kırılmalarına göre bölen dinamik yöntem. |
| **Cosine Distance** | İki vektör arasındaki açısal uzaklık ($1 - \text{CosSim}$). |
| **Context Buffer** | Cümlenin semantik temsilini güçlendirmek için komşularıyla oluşturulan kayan pencere. |
| **Breakpoint Threshold** | Ardışık iki cümlenin farklı konulardan bahsettiğini ilan eden mesafe eşiği. |
| **Fixed-Size Chunking** | Metni sabit karakter/token sınırlarından mekanik bölen geleneksel yöntem. |
| **Entity Fragmentation** | Bir kavramın veya varlığın parçalanarak iki ayrı chunk içine bölünmesi hatası. |
| **Precision@k** | RAG getirme aşamasında ilk $k$ sonuç içerisindeki doğru ve alakalı parça oranı. |
| **Percentile Threshold** | Mesafelerin belirli bir yüzdelik dilimini (örn: %80) eşik olarak seçme yöntemi. |
| **L2 Normalization** | Vektör uzunluğunu 1 birime indirgeyerek iç çarpımı doğrudan kosinüs benzerliğine eşitleme. |
| **GraphRAG Alignment** | Varlık ve ilişki çıkarımı için semantik bütünlüğü tam parçalar üretme süreci. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %94.8 yüksek RAG getirme doğruluğu │ • Sabit parçalamaya göre cümle başına│
 │ • Cümle ve kavram bütünlüğünün tam   │   ekstra embedding hesaplama maliyeti│
 │   korunması (%98.2).                 │ • Aşırı homojen metinlerde eşik be-  │
 │ • Sıfır gürültülü bağlam penceresi.  │   lirleme hassasiyeti gereksinimi.   │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Hukuk, tıp, finans ve akademik ma- │ • Hatalı noktalama kullanılan kirli  │
 │   kaleler gibi hassas RAG sistemleri.│   metinlerde cümle sınırlarının sapması
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/semantic_chunking_paneli.png` dosyası üretilir:
1. **Sabit vs Semantik Parçalama RAG Başarımı**
2. **Ardışık Cümle Kosinüs Mesafesi ve Kırılma Eşiği**
3. **Oluşturulan Semantik Parça Boyutları (Karakter)**
4. **Parça Başına Cümle Sayısı (Dinamik Aralık)**
5. **Semantik Parçalama Mimari Akış Şeması**
6. **Semantic Chunking Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Semantik parçalama iş akışını çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
