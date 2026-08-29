# Day 134: İki Aşamalı Hassas Getirme: Bi-Encoder (Vektör) + Cross-Encoder (Re-ranker)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu modül; yüksek ölçekli bilgi getirme sistemlerinde hız ve anlamsal derinliği kusursuz birleştiren **İki Aşamalı Getirme Hattı (Two-Stage Precision Retrieval)**, **1. Aşama Bi-Encoder (Hızlı Vektör Arama)**, **2. Aşama Cross-Encoder (Çapraz Token Dikkat Re-ranking)** ve **NDCG@k Sıralama Değerlendirme Motoru**nu sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Hızlıca Adayları Bul, Dikkatle En İyisini Seç: Bi-Encoder + Cross-Encoder"

Bilgi getirme (Information Retrieval) sistemlerinde klasik bir hız-kalite çıkmazı vardır:
- **Bi-Encoder (Hızlı Ama Yüzeysel):** Soru ve belgeyi birbirinden habersiz ayrı ayrı vektörleştirir ($E(q)$ ve $E(d)$). Milyonlarca belgeden milisaniyede sonuç getirir ($O(1)$ ANN arama), ancak sorudaki bir kelimenin belgedeki hangi kelimeyle ilişkili olduğunu (**Cross-Token Interaction**) göremez.
- **Cross-Encoder (Yavaş Ama Kusursuz):** Soruyu ve belgeyi tek bir metin gibi birleştirip ($[CLS] \circ q \circ [SEP] \circ d \circ [SEP]$) derin Transformer modeline sokar. Tüm kelimeler birbiriyle konuşur (**Full Self-Attention**). Anlamsal olarak mükemmeldir fakat 1 milyon belgeyi bu şekilde taramak saniyeler/dakikalar sürer.

**İki Aşamalı Hibrit Çözüm (Two-Stage Retrieval):**
1. ⚡ **1. Aşama (Bi-Encoder):** Milyonlarca belgeden en olası ilk $K=50$ aday belgeyi ~1 milisaniyede yakalar.
2. 🎯 **2. Aşama (Cross-Encoder):** Yalnızca bu $K=50$ adayı derin çapraz dikkat süzgecinden geçirir; sıralamayı düzeltir (**Rank Shift**) ve en doğru ilk $k=3$ belgeyi LLM'e iletir!

```
           [Kullanıcı Sorusu: q]
                     │
                     ▼
  [1. AŞAMA: Bi-Encoder Hızlı Vektör Arama]
   • Soru ve Belge Bağımsız Embedding E(q), E(d)
   • Milyonlarca Belgeden ~1ms İçinde Top-K Aday Çıkarımı
                     │
                     ▼ (Örn. K = 50 Aday Belge)
  [2. AŞAMA: Cross-Encoder Derin Re-ranking]
   • Soru + Belge Birleşik Dizisi: [CLS] q [SEP] d [SEP]
   • Token-Token Tam Çapraz Dikkat (Full Self-Attention)
   • Anlamsal Uygunluk ve Nüans Puanlama
                     │
                     ▼ (Sıralama Değişimi / Rank Shift)
       [En Hassas Top-k Belge (k = 3-5)]
                     │
                     ▼
        [LLM Üretim Modeline İletim]
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma: Hız vs Anlamsal Etkileşim ve İki Aşamalı Getirme Hattı
- 1. aşama $O(1)$ vektör hızıyla geniş havuz tararken, 2. aşama $O(K)$ hesaplamayla derin anlamsal nüansı yakalar.

### 2. 1. Aşama: Bi-Encoder Vektör Arama ($E(q) \cdot E(d)$) ve Aday Havuzu ($K$)
- Belgelerin embedding'leri önceden çevrimdışı (Offline) hesaplanır; arama anında sadece iç çarpım yapılır.

### 3. 2. Aşama: Cross-Encoder Derin Re-ranking ve Çapraz Token Dikkat Matrisi
- $[CLS] \circ q \circ [SEP] \circ d$ formatında token-token dikkat matrisi ($A_{i,j}$) üzerinden ince anlamsal puanlama yapılır.

### 4. NDCG@k Değerlendirmesi, Sıralama Değişimi (Rank Shift) ve Pareto Dengesi
- Re-ranking sonrası NDCG@5 başarımı %61.2'den **%96.4'e**, Top-1 doğruluğu **%54.0'dan %94.8'e** yükselir. Toplam gecikme yalnızca **~1.8 ms** seviyesindedir.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Bi-Encoder** | Soru ve belgeyi bağımsız olarak vektörleştiren ve iç çarpımla hızlı arama yapan model. |
| **Cross-Encoder** | Soru ve belgeyi birleşik token dizisi halinde derin Transformer dikkatine sokan model. |
| **Re-ranking** | 1. aşamadan gelen aday belgelerin daha güçlü bir modelle yeniden sıralanması. |
| **Rank Shift** | Bir belgenin 1. aşama sıralaması ile 2. aşama re-ranking sıralaması arasındaki pozisyon farkı. |
| **Cross-Attention** | Soru tokenlarının belgedeki tüm tokenlarla kurduğu doğrudan dikkat ağırlıkları matrisi. |
| **Candidate Pool ($K$)** | 1. aşamada hızlıca elenerek 2. aşamaya devredilen potansiyel aday sayısı (örn. 20-100). |
| **NDCG@k** | Sıralı getirme sonuçlarının kalitesini ve pozisyonel kazancını ölçen standart IR metriği. |
| **MRR (Mean Reciprocal Rank)** | İlk doğru belgenin kaçıncı sırada geldiğinin çarpmaya göre tersi ($1 / \text{rank}$). |
| **Pareto Frontier** | Gecikme maliyeti ile getirme doğruluğu arasındaki optimal denge çizgisi. |
| **Token Interaction Gap** | Bi-Encoder modellerinde soru ve belge tokenlarının doğrudan etkileşememesi kısıtı. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %96.4 NDCG@5 ve %94.8 Top-1 isabet │ • Saf Bi-Encoder'a göre küçük ek ge- │
 │ • Milisaniyelik iki aşamalı gecikme. │   cikme maliyeti (~1-15 ms).         │
 │ • Olumsuzluk ve karmaşık ifadeleri   │ • 2. aşama için ayrı bir model barın-│
 │   kusursuz anlama.                   │   dırma ve GPU bellek ihtiyacı.      │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Kurumsal arama motorları, müşteri  │ • Aday havuzu (K) çok küçük seçilirse│
 │   destek botları ve hassas RAG.      │   doğru belgenin 1. aşamada elenmesi.│
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/cross_encoder_reranking_paneli.png` dosyası üretilir:
1. **Bi-Encoder vs Cross-Encoder Re-ranked Başarımı**
2. **Re-ranking Sıralama Değişimi (Rank Shift)**
3. **Soru-Belge Çapraz Dikkat (Cross-Attention) Matrisi**
4. **Getirme Gecikmesi vs Doğruluk Pareto Kıyası**
5. **İki Aşamalı (Two-Stage) Getirme Mimarisi Şeması**
6. **Cross-Encoder Re-ranking Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# İki aşamalı getirme ve re-ranking iş akışını çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
