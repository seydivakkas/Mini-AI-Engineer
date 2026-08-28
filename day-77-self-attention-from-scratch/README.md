# Day 77: Sıfırdan Scaled Dot-Product & Multi-Head Self-Attention Mekanizması

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Attention: Scaled_Dot_Product](https://img.shields.io/badge/Attention-Multi--Head_Self--Attention-orange.svg?style=flat-square)](#matematiksel-formülasyon)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_self_attention.py)

Modern Derin Öğrenme ve Transformer devriminin (BERT, GPT, Vision Transformer - ViT, LLaMA) kalbinde yer alan **Ölçekli Nokta Çarpım Dikkat Mekanizması (Scaled Dot-Product Attention)** ve **Çok Kafalı Öz Dikkat (Multi-Head Self-Attention - MHSA)** bloklarını harici kütüphane kullanmadan, saf PyTorch tensör operasyonlarıyla sıfırdan inşa ediyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Geleneksel Derin Öğrenme mimarileri (CNN ve RNN/LSTM) görsel veya metinsel diziler arasındaki uzun menzilli ilişkileri kurarken ciddi yapısal sınırlamalara sahiptir:

1. **Sabit Yol Uzunluğu ($O(1)$ Path Length):**
   Bir RNN'de dizinin 1. elemanı ile 1000. elemanı arasında bilgi aktarmak için 1000 sıralı adım gerekir (ve gradyanlar yok olur). CNN'de ise iki pikselin etkileşebilmesi için onlarca evrişim katmanından geçip alıcı alanın (Receptive Field) büyümesi beklenir. **Self-Attention'da her token diğer tüm token'larla tek bir matris çarpımıyla $O(1)$ adımda doğrudan etkileşir.**
2. **Dinamik ve Girdiye Bağımlı Ağırlıklar (Data-Dependent Routing):**
   CNN'deki evrişim filtreleri ($3 \times 3$ çekirdekler) eğitim bittikten sonra sabittir; her görsele aynı ağırlık filtresi uygulanır. Self-Attention'da ise dikkat ağırlıkları ($A = \text{Softmax}(QK^\top / \sqrt{d_k})$) **girdinin o anki içeriğine göre dinamik olarak hesaplanır.**
3. **Çoklu Temsil Alt Uzayları (Multi-Head Diversity):**
   Tek bir dikkat başı yalnızca tek bir ilişki türüne (örneğin sadece nesnenin kenarlarına) odaklanabilir. $H$ adet bağımsız dikkat başı oluşturulduğunda; Baş 1 doku ilişkilerine, Baş 2 renk uyumuna, Baş 3 ise küresel simetriye odaklanarak zengin bir temsil uzayı oluşturur.

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Gradyan Doygunluğu ve Yok Olması (Softmax Saturation Darboğazı):**
  İki $d_k$ boyutlu rastgele vektörün nokta çarpımının varyansı $d_k$'dır. Eğer $d_k = 64$ veya $128$ gibi büyük seçilirse, nokta çarpım değerleri aşırı büyür ve Softmax fonksiyonunu türevinin $0$'a yakın olduğu doygunluk bölgesine iter. **$\frac{1}{\sqrt{d_k}}$ ile ölçekleme varyansı $1.0$'a sabitleyerek gradyanların gürbüz akmasını sağlar.**
- **Sıralı İşleme (Sequential Processing) Darboğazı:**
  RNN/LSTM ağları $t$ anındaki durumu hesaplamak için $t-1$'i beklemek zorundadır ve GPU'da paralelleştirilemez. Self-Attention tüm diziyi tek bir tensör matris çarpımı (`torch.matmul`) olarak eşzamanlı işler; GPU çekirdeklerini %100 doldurur.
- **Yerel Alıcı Alan Sınırlaması (Local Receptive Field):**
  Görselin sol üst köşesindeki bir yama (patch) ile sağ alt köşesindeki bir yama ilk katmandan itibaren küresel dikkat bağı kurabilir.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Karesel Bellek ve Zaman Karmaşıklığı ($O(N^2)$ Memory Wall):**
  $N$ adet token için dikkat matrisi $N \times N$ boyutundadır. $N = 1000$ için $1.000.000$ eleman iken, $N = 50.000$ için $2.5 \times 10^9$ eleman olur ve standart GPU belleğini (VRAM) anında tüketir. *(Çözüm: FlashAttention, Swin Window Attention).*
- **Tümevarımsal Yanlılık Eksikliği (Lack of Inductive Bias):**
  CNN'ler doğuştan "yakındaki pikseller birbiriyle ilişkilidir" (Translation Equivariance & Locality) ön kabulüne sahiptir. Self-Attention'da hiçbir ön kabul yoktur; pozisyonel kodlama (Positional Encoding) verilmezse token'ların sırasını bile ayırt edemez. Bu yüzden **büyük veri kümelerinde ön eğitim (Pre-training)** gerektirir.

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| Mekanizma | Zaman Karmaşıklığı | Bellek Karmaşıklığı | Küresel Bağlam (Global Context) | Paralelleşme |
|---|---|---|---|---|
| **Multi-Head Self-Attention (Bizim Mimari)** | $O(N^2 \cdot d)$ | $O(N^2 + N \cdot d)$ | ⭐⭐⭐⭐⭐ ($O(1)$ adımda) | ⭐⭐⭐⭐⭐ (Tam Paralel GPU) |
| **FlashAttention-2 (IO-Aware)** | $O(N^2 \cdot d)$ | **$O(N \cdot d)$ (SRAM Tiling)** | ⭐⭐⭐⭐⭐ ($O(1)$ adımda) | ⭐⭐⭐⭐⭐ (Hardware Native) |
| **Evrişimsel Katman (CNN 3x3)** | $O(N \cdot K^2 \cdot d)$ | $O(N \cdot d)$ | ⭐⭐ (Sadece yerel pencere) | ⭐⭐⭐⭐⭐ (Paralel) |
| **LSTM / GRU (Tekrarlayan Ağ)** | $O(N \cdot d^2)$ | $O(N \cdot d)$ | ⭐⭐⭐ ($O(N)$ adımda kayıplı) | ⭐ (Sıralı, GPU blokajı) |
| **State Space Models (Mamba / S4)** | **$O(N \cdot d)$ (Lineer)** | **$O(N \cdot d)$** | ⭐⭐⭐⭐ (Seçici Durum Uzayı) | ⭐⭐⭐⭐ (Paralel Scan) |

---

## 📐 Matematiksel Formülasyon

### 1. Girdi Temsili ve Q, K, V Doğrusal Projeksiyonları
Girdi dizisi $X \in \mathbb{R}^{B \times N \times D_{\text{model}}}$ için öğrenilebilir ağırlık matrisleri $W_Q, W_K, W_V \in \mathbb{R}^{D_{\text{model}} \times D_{\text{model}}}$ ile:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

### 2. Ölçekli Nokta Çarpım Dikkati (Scaled Dot-Product Attention)
$Q, K, V$ tensörleri $H$ adet başa bölünür ($d_k = D_{\text{model}} / H$). Her baş için dikkat skoru:

$$S = \frac{Q K^\top}{\sqrt{d_k}} \in \mathbb{R}^{B \times H \times N \times N}$$

$$\text{Attention}(Q, K, V) = \text{Softmax}\left( S + M \right) V$$

Burada $M$ isteğe bağlı dikkat maskesidir ($M_{i, j} = -\infty$ ile nedensel maskeleme veya dolgu maskelemesi yapılabilir).

### 3. Çoklu Kafaların Birleştirilmesi ve Çıkış Projeksiyonu (W_O)
Her başın ürettiği $\text{head}_h \in \mathbb{R}^{B \times N \times d_k}$ tensörleri son boyutta birleştirilir (Concat) ve $W_O \in \mathbb{R}^{D_{\text{model}} \times D_{\text{model}}}$ ile çarpılır:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_H) W_O$$

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama |
|---|---|---|
| **Query ($Q$)** | *Sorgu Vektörü* | Bir token'ın diğer tüm token'lardan ne tür bilgi aradığını temsil eden öznitelik vektörü. |
| **Key ($K$)** | *Anahtar Vektörü* | Bir token'ın sorgulara cevap verebilmek için sunduğu kimlik/içerik öznitelik vektörü. |
| **Value ($V$)** | *Değer Vektörü* | Sorgu-Anahtar eşleşmesi gerçekleştiğinde aktarılacak olan asıl anlamsal bilgi yükü. |
| **Scale Factor ($\frac{1}{\sqrt{d_k}}$)** | *Ölçekleme Faktörü* | Nokta çarpım büyüklüğünü normalize ederek Softmax gradyanlarının yok olmasını engelleyen katsayı. |
| **Multi-Head Attention** | *Çok Kafalı Dikkat* | Temsil boyutunu alt uzaylara bölerek modelin aynı anda farklı ilişkileri öğrenmesini sağlayan mekanizma. |
| **Attention Mask** | *Dikkat Maskesi* | Dolgu token'larının (Padding) veya gelecekteki token'ların (Causal) dikkat almasını engelleyen matris. |
| **Head Diversity** | *Başlar Arası Çeşitlilik* | Farklı dikkat başlarının birbirini tekrar etmeyip farklı desenler öğrenme derecesi (Kosinüs mesafesi). |
| **Softmax Saturation** | *Softmax Doygunluğu* | Girdi logitlerinin aşırı büyümesi sonucu Softmax çıktısının One-Hot'a yaklaşması ve gradyanın sıfırlanması. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Uzun menzilli global bağlam yakalama; Mükemmel GPU paralelleşmesi; Zengin çoklu baş (Multi-Head) temsili. |
| **Weaknesses (Zayıf Yönler)** | $O(N^2)$ bellek duvarı; Konumsal tümevarım yanlılığının olmaması (Positional Encoding zorunluluğu). |
| **Opportunities (Fırsatlar)** | Vision Transformer (ViT), Çok Modlu (VLM) ve LLM mimarilerinin ana omurgası; FlashAttention optimizasyonları. |
| **Threats (Tehditler)** | Çok uzun dizilerde (ör. $4K \times 4K$ yüksek çözünürlüklü görseller) donanım belleğinin kilitlenmesi. |

---

## 💻 Üretim Seviyesinde Uygulama Mimarisi

Kaynak kodları [`day-77-self-attention-from-scratch/`](.) dizininde yer almaktadır:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class OlcekliNoktaCarpimDikkat(nn.Module):
    """Attention(Q, K, V) = Softmax((Q * K^T) / sqrt(d_k) + M) * V"""
    def __init__(self, dropout_orani: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_orani) if dropout_orani > 0.0 else None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        d_k = q.size(-1)
        skorlar = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            skorlar = skorlar.masked_fill(mask == 0, -1e9)
        agirliklar = F.softmax(skorlar, dim=-1)
        if self.dropout is not None:
            agirliklar = self.dropout(agirliklar)
        return torch.matmul(agirliklar, v), agirliklar
```

---

## 📊 Deneysel Benchmark ve Doğrulama Sonuçları

`ana_akis.py` çalıştırılarak elde edilen analitik çıktılar:

```text
======================================================================
  Dikkat Başı   | Ortalama Entropi (Bit) |  Ort. Dikkat Mesafesi 
======================================================================
     Baş 1      |         2.718          |       5.25 token      
     Baş 2      |         2.728          |       5.28 token      
     Baş 3      |         2.716          |       5.33 token      
     Baş 4      |         2.722          |       5.35 token      
----------------------------------------------------------------------
📌 Başlar Arası Çeşitlilik Skoru (Kosinüs Mesafesi): 0.0254 / 1.000
📌 1/√d_k Ölçekleme Öncesi Entropi: 1.125 -> Ölçekleme Sonrası: 2.428
======================================================================
```

### 🔑 Analiz ve Çıkarımlar
1. **Ölçekleme Koruması:** $\frac{1}{\sqrt{d_k}}$ ölçeklemesi uygulanmadığında entropi $1.125$ seviyesine çökmekte (Softmax doygunluğu), ölçekleme uygulandığında ise **$2.428$ seviyesinde dengeli ve zengin bir dağılım** korunmaktadır.
2. **Gradyan Sağlığı:** Geriye yayılımda $W_Q, W_K, W_V, W_O$ matrislerinin tümü stabil ve sıfırdan farklı gradyanlar üretmiştir.
3. **Birim Test Güvencesi:** [`testler/test_self_attention.py`](testler/test_self_attention.py) altındaki **8/8 birim test %100 PASSED (3.72s)** ile doğrulanmıştır.

---

## 🎨 6 Panelli Teşhis Panosu

Üretilen yüksek çözünürlüklü teşhis panosu [`ciktilar/self_attention_paneli.png`](ciktilar/self_attention_paneli.png) konumundadır:

1. **MHSA Matematiksel Hesaplama Akışı:** $Q, K, V$ projeksiyonlarından çıkış katmanına tam akış.
2. **Multi-Head Dikkat Isı Haritası (Heatmaps):** 4 başın $16 \times 16$ dikkat matrisleri grid görseli.
3. **$\sqrt{d_k}$ Ölçeklemenin Softmax Doygunluğuna Etkisi:** Ölçekli vs ölçeksiz dağılım eğrileri.
4. **Başların Uzamsal Alıcı Alanı:** Lokal vs global token dikkat mesafesi çubukları.
5. **Baş Entropisi ve Çeşitlilik:** Her başın bilgi entropisi ve başlar arası kosinüs mesafesi.
6. **MHSA Mimari SWOT Karar Matrisi:** Mühendislik karar tablosu.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** Büyük modellerde (LLaMA-3, Mistral) kullanılan ve Key/Value baş sayısını Query baş sayısından daha az tutarak bellek kullanımını dramatik şekilde düşüren **Grouped-Query Attention (GQA)** mekanizmasını sıfırdan yazınız.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA): num_q_heads > num_kv_heads"""
    def __init__(self, d_model: int = 64, num_q_heads: int = 8, num_kv_heads: int = 2):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_q_heads // num_kv_heads
        self.d_k = d_model // num_q_heads

        self.w_q = nn.Linear(d_model, num_q_heads * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.w_q(x).view(b, n, self.num_q_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(b, n, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, n, self.num_kv_heads, self.d_k).transpose(1, 2)

        # KV başlarını Q sayısına genişlet (Repeat KV)
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        skorlar = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        agirliklar = F.softmax(skorlar, dim=-1)
        cikti = torch.matmul(agirliklar, v).transpose(1, 2).contiguous().view(b, n, -1)
        return self.w_o(cikti)
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Nokta çarpım dikkat mekanizmasında neden $\frac{1}{\sqrt{d_k}}$ ile bölme yapılır? Eğer bu ölçekleme yapılmazsa model eğitiminde tam olarak ne gibi matematiksel ve sayısal problemler meydana gelir?

> **Mentor Cevabı:**
> 1. **Varyans Patlaması:** $q$ ve $k$ vektörlerinin elemanlarının sıfır ortalamalı ve birim varyanslı ($\mu=0, \sigma^2=1$) bağımsız rastgele değişkenler olduğunu varsayalım. İki vektörün nokta çarpımı $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ ifadesinin ortalaması $0$, varyansı ise tam olarak $d_k$'dır.
> 2. **Softmax Doygunluğu (Saturation):** Eğer $d_k = 64$ ise nokta çarpım standart sapması $\sqrt{64} = 8$ olur. Logit değerleri $+15, -20$ gibi çok büyük sayılara ulaştığında, Softmax çıktısı en büyük eleman için $1.0$'a, diğer tüm elemanlar için $0.0$'a yaklaşarak One-Hot vektöre dönüşür.
> 3. **Gradyan Yok Olması (Vanishing Gradient):** Softmax fonksiyonunun türevi $\frac{\partial S_i}{\partial z_j} = S_i (\delta_{ij} - S_j)$ şeklindedir. $S_i \approx 1$ ve $S_j \approx 0$ olduğunda türev neredeyse sıfır ($0.0$) olur. Bu durumda geriye doğru hiçbir gradyan akmaz ve $W_Q, W_K$ katmanlarının eğitimi tamamen durur. $\frac{1}{\sqrt{d_k}}$ ile bölmek varyansı tekrar $1.0$'a indirerek bu felaketi engeller.

---

## 📜 Lisans & Metaveri

```text
/*
 * Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101 Günlük Yapay Zeka, Bilgisayarlı Görü ve MLOps Mühendisliği
 * Özel Lisans — Tüm Hakları Saklıdır.
 */
```
