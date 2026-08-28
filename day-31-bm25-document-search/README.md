# Day 31: BM25 Leksikal Belge Arama Motoru (BM25 Document Search)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557C.svg?style=flat-square)](https://matplotlib.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; Bilgi Erişimi (Information Retrieval), Arama Motorları (Elasticsearch, Lucene, OpenSearch) ve Modern RAG (Retrieval-Augmented Generation) mimarilerinin vazgeçilmez leksikal omurgası olan **Okapi BM25 (Best Matching 25)** arama motorunun sıfırdan ters indeks (inverted index) mimarisiyle geliştirilmiş tam kapsamlı bir uygulamasıdır.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. TF-IDF'in Sınırları ve BM25'in Getirdiği Yenilikler
Klasik TF-IDF (Term Frequency - Inverse Document Frequency) modelinde iki kritik problem bulunur:
1. **Doğrusal Terim Frekansı Büyümesi (Linear TF Scaling):** Bir belgede aranan kelime 1 kez yerine 100 kez geçtiğinde, klasik TF-IDF puanı neredeyse 100 katına çıkar. Ancak bir belgede "Python" kelimesinin 10 kez geçmesi ile 100 kez geçmesi arasında bilgi içeriği bakımından 10 kat fark yoktur (azalan verimler yasası).
2. **Belge Uzunluğu Adaletsizliği:** Uzun belgeler doğal olarak daha fazla kelime içerdiği için klasik modellerde haksız avantaj kazanır.

**BM25'in Getirdiği Çözümler:**
- **Asimptotik TF Doygunluğu ($k_1$):** Terim sıklığı arttıkça puanın logaritmik/asimptotik olarak bir tavana yaklaşmasını sağlar.
- **Dinamik Uzunluk Normalizasyonu ($b$):** Belgenin uzunluğunu korpusun ortalama uzunluğuna ($\text{avgdl}$) oranlayarak adil puanlama yapar.

```
                  ┌──────────────────────────────────────────────────────────┐
                  │                 GİRİŞ METİN KORPUSU                      │
                  └────────────────────────────┬─────────────────────────────┘
                                               │
                                               ▼
                  ┌──────────────────────────────────────────────────────────┐
                  │  TOKENİZASYON & NORMALİZASYON (METİN ÖN İŞLEME)          │
                  │  - Küçük harf, Noktalama Temizliği, Stop-words Filtresi  │
                  └────────────────────────────┬─────────────────────────────┘
                                               │
                                               ▼
                  ┌──────────────────────────────────────────────────────────┐
                  │  TERS İNDEKS MİMARİSİ (INVERTED INDEX)                   │
                  │  - Terim -> {Doc_ID_1: TF, Doc_ID_2: TF, ...}           │
                  │  - avgdl (Ortalama Belge Uzunluğu) Takibi                │
                  └────────────────────────────┬─────────────────────────────┘
                                               │
                                               ▼
                  ┌──────────────────────────────────────────────────────────┐
                  │  OKAPI BM25 PUANLAMA MOTORU                              │
                  │  - Okapi IDF (Ters Belge Frekansı)                       │
                  │  - k1 (TF Doygunluk) & b (Uzunluk Normalizasyon) Hesabı  │
                  └────────────────────────────┬─────────────────────────────┘
                                               │
                                               ▼
                  ┌──────────────────────────────────────────────────────────┐
                  │  SIRALANMIŞ EN İYİ BELGELER (TOP-K RETRIEVAL ÇIKTISI)    │
                  └──────────────────────────────────────────────────────────┘
```

---

#

---

### 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Ters Doküman Frekansı (IDF) ve terim sıklığı doygunluğu ile anahtar kelime tabanlı hassas metin aramasını milisaniyeler içinde yapmak için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Spesifik ürün kodları, seri numaraları ve teknik terimlerin vektör aramasında kaybolmasını önler; sıfır eğitim maliyetiyle güçlü arama sağlar.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Eşanlamlılık (synonymy) ve bağlamsal anlamsallığı (polysemy) anlayamaz; kelime birebir geçmiyorsa sonuç bulamaz.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- Dense Semantic Retrieval (Embedding-based), Splade veya Hibrit Arama (Hybrid Search).

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **BM25 Algoritması** | *Best Matching 25* | Klasik bilgi erişiminde (Information Retrieval) terim sıklığı ve ters belge sıklığını belge uzunluğu normalizasyonuyla birleştiren sıralama fonksiyonu. |
| **Terim Sıklığı Doygunluğu ($k_1$)** | *Term Frequency Saturation* | Bir kelimenin belgede çok fazla geçmesinin skor üzerindeki etkisini asimptotik olarak sınırlandıran hiperparametre. |
| **Belge Uzunluğu Cezalandırması ($b$)** | *Document Length Normalization* | Uzun belgelerin yalnızca çok kelime içerdikleri için haksız yere yüksek skor almasını önleyen parametre. |
| **Ters Belge Sıklığı (IDF)** | *Inverse Document Frequency* | Tüm külliyatta nadir geçen terimlere daha yüksek bilgi ağırlığı veren logaritmik formül. |

---

## 2. Matematiksel Formülasyon

#### A. Okapi BM25 Uygunluk Skoru (Score)
Bir $Q = \{q_1, q_2, \dots, q_n\}$ sorgusu ve bir $D$ belgesi için toplam skor:

$$\text{Score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Burada:
- $f(q, D)$: $q$ teriminin $D$ belgesindeki frekansı (Term Frequency).
- $|D|$: $D$ belgesinin toplam kelime sayısı (uzunluğu).
- $\text{avgdl}$: Koleksiyondaki tüm belgelerin ortalama kelime uzunluğu.
- $k_1$: Terim frekansı doygunluk parametresi ($1.2 \le k_1 \le 2.0$, standart $1.5$).
- $b$: Belge uzunluğu normalizasyon/ceza katsayısı ($0 \le b \le 1$, standart $0.75$).

#### B. Düzeltilmiş Okapi IDF (Inverse Document Frequency)
Negatif değer üretmeyen güvenli Okapi IDF formülü:

$$\text{IDF}(q) = \ln \left( \frac{N - n(q) + 0.5}{n(q) + 0.5} + 1 \right)$$

- $N$: Korpus içindeki toplam belge sayısı.
- $n(q)$: $q$ terimini içeren toplam belge sayısı.

---

### 3. $k_1$ ve $b$ Hiperparametrelerinin Rolü

| Parametre | Standart Değer | Etkisi |
|---|---|---|
| **$k_1$ (TF Doygunluğu)** | `1.5` | Bir kelimenin belgede tekrar etme sayısının skora olan katkı hızını belirler. $k_1 \to 0$ olursa TF etkisi kaybolur (sadece var/yok olur); $k_1 \to \infty$ olursa klasik doğrusal TF'e yaklaşır. |
| **$b$ (Uzunluk Cezası)** | `0.75` | Belge uzunluğunun normalize edilme derecesidir. $b = 1$ tam uzunluk normalizasyonu uygular; $b = 0$ uzunluk farklarını tamamen görmezden gelir. |

---

## 📊 Arama Performansı ve Analiz Sonuçları

Örnek 10 kurumsal/teknik belge üzerinde yapılan leksikal arama sonuçları:

| Sıra | Belge ID | Belge Başlığı | BM25 Skoru | Ana Katkı Sağlayan Terimler |
|---|---|---|---|---|
| **1** | `DOC-004` | Vektör Veritabanları ve Semantik Arama | **4.8210** | `vektör` (2.41), `veritabanı` (2.41) |
| **2** | `DOC-005` | RAG Mimarisi | **3.9450** | `rag` (2.05), `mimari` (1.89) |
| **3** | `DOC-009` | Hibrit Arama ve RRF | **3.1240** | `arama` (1.65), `semantik` (1.47) |

---

## 🛠️ Dizin Yapısı

```
day-31-bm25-document-search/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca indeksleme ve BM25 sorgu akışı
├── README.md                        # Detaylı teorik ve matematiksel dokümantasyon (220+ Satır)
├── src/
│   ├── __init__.py
│   ├── tokenlestirici.py            # Metin temizleme, küçük harf ve stop-words motoru
│   ├── ters_indeks.py               # Postings list ve ters indeks veri yapısı
│   ├── bm25_motoru.py               # Okapi BM25 puanlama ve IDF hesaplayıcı
│   ├── arama_sunucusu.py            # Belge yönetimi ve parametre analiz motoru
│   └── gorsellestirici.py           # 6 panelli teşhis panosu (Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_bm25_search.py          # 6 adet kapsamlı birim test
└── ciktilar/
    └── bm25_arama_paneli.png        # 6 panelli BM25 analiz panosu görseli
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

**Görev:** `src/bm25_motoru.py` içerisine sorgudaki tüm kelimelerin tam eşleştiği belgeleri öne çıkaran bir **"Exact Phrase Bonus / Zorunlu Terim Çarpanı"** eklemek.

**Çözüm:**
```python
def phrase_bonus_ile_puanla(self, sorgu_tokenlari: List[str], doc_id: str, bonus_carpani: float = 1.25) -> float:
    skor, _ = self.belge_puani_hesapla(sorgu_tokenlari, doc_id)
    belge_tokenlar = self.indeks.belgeler[doc_id]["tokenlar"]
    # Sorgudaki tüm tokenlar belgede var mı?
    if set(sorgu_tokenlari).issubset(set(belge_tokenlar)):
        skor *= bonus_carpani
    return float(skor)
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Modern RAG (Retrieval-Augmented Generation) sistemlerinde ve arama motorlarında yalnızca yoğun vektör gömmeleri (Dense Vector Search) kullanmak yerine neden **BM25 leksikal arama ile birleştirilmiş Hibrit Arama (Hybrid Search)** tercih edilir?

> **Cevap:**
> 1. **Nadir Terimler ve Spesifik Kodlar:** Dense vektör modelleri (örneğin Sentence Transformers veya OpenAI embeddings), kelimelerin anlamsal bağlamını yakalamakta mükemmeldir; ancak ürün kodları (ör. `GTX-4090`), model numaraları, tıbbi terimler veya özel isimler gibi korpusta nadir geçen leksikal anahtarlarda zayıf kalırlar.
> 2. **BM25'in Kesin Eşleşme Gücü:** BM25, tam kelime eşleşmesinde ve yüksek IDF'e sahip nadir teknik terimlerde rakipsizdir. Hibrit arama mimarisinde BM25 leksikal araması ile yoğun semantik vektör araması birleştirildiğinde hem anlamsal benzerlik hem de kesin kod/terim doğruluğu eşzamanlı olarak elde edilir.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
