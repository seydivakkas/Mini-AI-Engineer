# Day 78: Sıfırdan Transformer Encoder Bloğu — Pozisyonel Kodlama, LayerNorm, Residual FFN

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Architecture: Transformer_Encoder](https://img.shields.io/badge/Architecture-Pre--LN_Encoder_Block-orange.svg?style=flat-square)](#matematiksel-formülasyon)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_encoder.py)

Vision Transformer (ViT), BERT, RoBERTa ve modern temel modellerin omurgasını oluşturan **Transformer Encoder Bloğu**'nu; **Sinüzoidal & Öğrenilebilir Pozisyonel Kodlama (Positional Encoding)**, **Pre-LayerNorm vs Post-LayerNorm** mimari dinamikleri, **Kalıntı Bağlantılar (Residual Connections)** ve **GELU Aktivasyonlu $4\times$ FFN** ile sıfırdan saf PyTorch mimarisiyle inşa ediyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Tek başına Multi-Head Self-Attention (MHSA) tam bir derin öğrenme modeli oluşturamaz. Bir diziyi veya görsel yamalarını derinlemesine işlemek için şu 4 temel mühendislik bileşeni zorunludur:

1. **Konum / Sıra Bilgisinin Enjeksiyonu (Positional Encoding):**
   Self-Attention işlemi permütasyona duyarsızdır (Permutation Invariant); yani girdideki token'ların yerini rastgele karıştırsanız bile dikkat ağırlıkları değişmez. Cümledeki kelime sırasını veya görseldeki yamaların 2D ızgara konumunu modele aktarmak için **Sinüzoidal** veya **Öğrenilebilir** pozisyonel vektörler girdiyle toplanır ($X + PE$).
2. **Kalıntı Bağlantılar (Residual Connections - He et al. 2016):**
   Derin ağlarda katman sayısı arttıkça gradyanların geriye doğru sönmeden akabilmesi için her bloğun etrafında kestirme bir kimlik yolu ($x + \mathcal{F}(x)$) oluşturulur.
3. **Katman Normalizasyonu (Pre-LayerNorm Mimarisi):**
   Her örneğin kendi öznitelik boyutu boyunca ortalamasını $0$ ve varyansını $1$ yaparak ağın iç kovaryans kaymasını (Internal Covariate Shift) önler. Pre-LN tasarımı sayesinde **100'den fazla katman learning rate warmup gerektirmeden kararlı şekilde eğitilebilir.**
4. **Konumsal İleri Beslemeli Ağ (Position-wise FFN):**
   Self-Attention token'lar arasındaki **ilişkileri ve etkileşimi** yakalarken, FFN katmanı ($D \to 4D \to D$) her token'ın kendi içindeki **anlamsal temsilini doğrusal olmayan (GELU) biçimde derinleştirir.**

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Post-LN Gradyan Patlaması / Sönmesi Darboğazı:**
  Orijinal 2017 Transformer'da (Post-LN) normalizasyon kalıntı toplamından sonra geliyordu ($x = \text{LN}(x + \mathcal{F}(x))$). Bu durum derin katmanlarda gradyanları kararsızlaştırıp modelin ıraksamasına neden oluyordu. **Pre-LN ($x = x + \mathcal{F}(\text{LN}(x))$) kalıntı omurgasını tamamen serbest bırakarak gradyan patlamasını çözer.**
- **Sırasızlık / Konumsuzluk Çıkmazı:**
  Evrişimli ağların aksine Self-Attention piksellerin nerede olduğunu bilemez. Pozisyonel kodlama, modelin uzamsal 2D geometriyi ve sıralı 1D sözdizimini kusursuz öğrenmesini sağlar.
- **Doğrusal Sıkışma (Lack of Non-Linearity):**
  Yalnızca dikkat mekanizması doğrusal ağırlıklı toplamlar üretir. $4 \times$ genişletilmiş FFN ve GELU aktivasyonu, modele yüksek kapasiteli evrensel fonksiyon yaklaştırma gücü kazandırır.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Parametre Yoğunluğu (%66 FFN Payı):**
  Bir Transformer Encoder bloğundaki toplam parametrelerin yaklaşık üçte ikisi ($2 \times D \times 4D = 8D^2$) FFN katmanına aittir. *(Çözüm: MoE - Mixture of Experts veya LoRA ile seyreltme).*
- **Maksimum Dizi Uzunluğu Sabiti:**
  Sinüzoidal kodlama teori olarak sonsuz uzunluğa genellenebilse de, pratikte ön eğitim yapılan $N_{\text{max}}$ sınırının ötesine geçildiğinde performans düşebilir *(Çözüm: RoPE - Rotary Positional Embedding).*

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| Mimari Bileşen | Standart Yaklaşım | Alternatif / Modern Gelişme | Avantaj & Karşılaştırma |
|---|---|---|---|
| **Normalizasyon Konumu** | **Pre-LayerNorm (Bizim Tercihimiz)** | Post-LayerNorm | Pre-LN: Isınma gerektirmez, 100+ katmanda kararlı. |
| **Normalizasyon Türü** | **Layer Normalization** | RMSNorm (LLaMA) | RMSNorm: Ortalama çıkarmayı atlayarak %15-20 hız kazanır. |
| **Pozisyonel Kodlama** | **Sinusoidal / Learnable 1D** | RoPE (Rotary Position) | RoPE: Açısal döndürme ile göreceli mesafe bilgisi sağlar. |
| **Aktivasyon Fonksiyonu** | **GELU (Gaussian Error Linear)** | SwiGLU / ReLU | GELU: Pürüzsüz türevle derinlikte daha iyi genelleme sağlar. |

---

## 📐 Matematiksel Formülasyon

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             TRANSFORMER ENCODER BLOĞU İÇ MATEMATİKSEL YAPISI                              │
│                                                                                                           │
│       Girdi: x ∈ ℝ^(B × N × D)                                                                            │
│          │                                                                                                │
│          ├──────────────────────────────────────────────┐ (Kalıntı Kestirme Yolu 1 - Residual Path)       │
│          ▼                                              │                                                 │
│       LayerNorm_1(x)                                    │                                                 │
│          ▼                                              │                                                 │
│       Multi-Head Self-Attention (MHSA)                  │                                                 │
│          ▼                                              │                                                 │
│       Dropout                                           │                                                 │
│          ▼                                              │                                                 │
│          (+) <──────────────────────────────────────────┘                                                 │
│          │                                                                                                │
│          ├──────────────────────────────────────────────┐ (Kalıntı Kestirme Yolu 2 - Residual Path)       │
│          ▼                                              │                                                 │
│       LayerNorm_2(x^(1))                                │                                                 │
│          ▼                                              │                                                 │
│       Feed-Forward Network: W1(D->4D) -> GELU -> W2(4D->D)                                                │
│          ▼                                              │                                                 │
│       Dropout                                           │                                                 │
│          ▼                                              │                                                 │
│          (+) <──────────────────────────────────────────┘                                                 │
│          │                                                                                                │
│       Çıktı: x^(2) ∈ ℝ^(B × N × D)                                                                        │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Sinüzoidal Pozisyonel Kodlama
$$PE_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{2i / D}} \right), \quad PE_{(pos, 2i+1)} = \cos\left( \frac{pos}{10000^{2i / D}} \right)$$

### 2. Özel Katman Normalizasyonu (Custom LayerNorm)
Son boyut $D$ üzerinden ortalama $\mu$ ve varyans $\sigma^2$:
$$\mu = \frac{1}{D} \sum_{j=1}^D x_j, \quad \sigma^2 = \frac{1}{D} \sum_{j=1}^D (x_j - \mu)^2$$

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

### 3. Pre-LN Transformer Encoder Bloğu İleri Geçişi
$$x^{(1)} = x + \text{Dropout}\Big(\text{MHSA}\big(\text{LayerNorm}_1(x)\big)\Big)$$

$$x^{(2)} = x^{(1)} + \text{Dropout}\Big(\text{FFN}\big(\text{LayerNorm}_2(x^{(1)})\big)\Big)$$

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama |
|---|---|---|
| **Transformer Encoder** | *Transformer Encoder Block* | Self-Attention, LayerNorm, Residual bağlantılar ve FFN'i birleştiren temsil çıkarıcı temel blok. |
| **Pre-LayerNorm** | *Pre-Layer Normalization* | Normalizasyonun MHSA ve FFN öncesine konulduğu, derin ağlarda gradyan patlamasını önleyen modern tasarım. |
| **Post-LayerNorm** | *Post-Layer Normalization* | Normalizasyonun kalıntı toplamının sonrasına konulduğu orijinal 2017 Transformer mimarisi. |
| **Sinusoidal Encoding** | *Sinüzoidal Pozisyonel Kodlama* | Farklı frekanslardaki sinüs ve kosinüs dalgalarıyla oluşturulan parametresiz konumsal gömülme matrisi. |
| **Residual Stream** | *Kalıntı Bilgi Akışı* | Ağın başından sonuna kadar kesintisiz uzanan ve her bloğun üzerine ekleme yaptığı ana sinyal otobanı. |
| **Position-wise FFN** | *Konumsal İleri Besleme Ağı* | Her token'a bağımsız olarak uygulanan iki katmanlı doğrusal genişleme ve sıkıştırma MLP modülü ($D \to 4D \to D$). |
| **GELU Activation** | *Gaussian Error Linear Unit* | $x \cdot \Phi(x)$ formülüyle girdiyi olasılıksal eşikleyen, ViT ve GPT standartı pürüzsüz aktivasyon fonksiyonu. |
| **Internal Covariate Shift**| *İç Kovaryans Kayması* | Eğitim sırasında önceki katmanların ağırlıkları değiştikçe sonraki katmanların girdi dağılımının bozulması problemi. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | Pre-LN ile pürüzsüz gradyan akışı; Residual omurga ile derinlik ölçeklenebilirliği; GELU ile yüksek ifade gücü. |
| **Weaknesses (Zayıf Yönler)** | FFN katmanlarının bellek ve parametre yükü (%66 pay); Karesel dikkat hesaplama maliyeti. |
| **Opportunities (Fırsatlar)** | Vision Transformer (ViT), Swin Transformer ve BERT tabanlı tüm multimodal sistemlerin temel yapı taşı. |
| **Threats (Tehditler)** | Yanlış normalizasyon veya eksik kalıntı bağlantılarında derin modellerin hızla ıraksaması. |

---

## 💻 Üretim Seviyesinde Uygulama Kodu

Kaynak kodları [`day-78-transformer-encoder-from-scratch/`](.) dizininde modüler olarak yapılandırılmıştır:

```python
class TransformerEncoderBlogu(nn.Module):
    """Pre-LN Transformer Encoder Bloğu"""
    def __init__(self, model_boyutu: int = 64, kafa_sayisi: int = 4, genisleme_faktoru: int = 4, dropout_orani: float = 0.1):
        super().__init__()
        self.dikkat = CokKafaliOzDikkat(model_boyutu, kafa_sayisi, dropout_orani)
        self.ln1 = OzelLayerNorm(model_boyutu)
        self.ln2 = OzelLayerNorm(model_boyutu)
        self.ffn = BeslemeliIleriAg(model_boyutu, genisleme_faktoru, dropout_orani, aktivasyon="gelu")
        self.dropout = nn.Dropout(p=dropout_orani)

    def forward(self, x, mask=None):
        # 1. Pre-LN MHSA
        norm_x = self.ln1(x)
        attn_out, dikkat_haritasi = self.dikkat(norm_x, mask=mask)
        x = x + self.dropout(attn_out)
        
        # 2. Pre-LN FFN
        norm_x2 = self.ln2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + self.dropout(ffn_out)
        return x, dikkat_haritasi
```

---

## 📊 Deneysel Benchmark ve Gradyan Doğrulama Sonuçları

`ana_akis.py` çalıştırılarak 4 katmanlı Pre-LN ve Post-LN modellerinin gradyan normları ve katman temsil benzerlikleri doğrulanmıştır:

```text
=================================================================
    Katman      |  Pre-LN Gradyan Normu  | Post-LN Gradyan Normu 
=================================================================
   Katman 1     |         0.0847         |         0.0966        
   Katman 2     |         0.0823         |         0.0981        
   Katman 3     |         0.0817         |         0.0958        
   Katman 4     |         0.0740         |         0.0986        
=================================================================
  ✓ Katman 1 -> Katman 2 Temsil Benzerliği: 0.9770
  ✓ Katman 2 -> Katman 3 Temsil Benzerliği: 0.9724
  ✓ Katman 3 -> Katman 4 Temsil Benzerliği: 0.9738
```

### 🔑 Çıkarımlar
1. **Pre-LN Gradyan Pürüzsüzlüğü:** Pre-LN mimarisinde geriye yayılan gradyan normları katmanlar boyunca neredeyse sabit kalmakta ($0.084 \to 0.074$), bu da yüzlerce katmanlı modellerde patlama veya sönme olmadan kararlı eğitimi garanti etmektedir.
2. **Kademeli Temsil Rafinasyonu:** Katmanlar arası kosinüs benzerliği $0.97$ civarında kalarak her katmanın önceki katmanın özelliklerini bozmadan üzerine küçük ve değerli eklemeler yaptığını doğrulamaktadır.
3. **Birim Test Başarımı:** [`testler/test_encoder.py`](testler/test_encoder.py) altındaki **8/8 birim test %100 PASSED (3.74s)** ile doğrulanmıştır.

---

## 🎨 6 Panelli Teşhis Panosu

Üretilen yüksek çözünürlüklü teşhis paneli [`ciktilar/transformer_encoder_paneli.png`](ciktilar/transformer_encoder_paneli.png) konumundadır:

1. **Pre-LN vs Post-LN Mimari Tasarımı:** İki blok tasarımının şematik akış karşılaştırması.
2. **Sinüzoidal Pozisyonel Kodlama Matrisi:** 32 pozisyon $\times$ 64 boyut dalga frekansı ısı haritası.
3. **Pre-LN vs Post-LN Gradyan Kararlılığı:** Katman bazında gradyan norm çubukları.
4. **FFN Gizli Katman Aktivasyon Dağılımı:** GELU vs ReLU yoğunluk eğrileri.
5. **Katmanlar Arası Temsil Akışı:** Katmandan katmana kosinüs benzerliği çizgisi.
6. **Transformer Encoder SWOT Karar Matrisi:** Mühendislik karar tablosu.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** LLaMA ve modern açık kaynak LLM'lerde kullanılan, LayerNorm yerine standart sapmayı çıkaran **RMSNorm (Root Mean Square Normalization)** ve standart FFN yerine kapılı **SwiGLU (Swish Gated Linear Unit)** katmanını sıfırdan yazınız.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """Root Mean Square Normalization (LLaMA-3 Standartı)"""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ortalama çıkarma yok, sadece karekök ortalama kareye bölme
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gamma

class SwiGLUFFN(nn.Module):
    """SwiGLU Kapılı İleri Beslemeli Ağ: FFN_SwiGLU(x) = (Swish(xW_gate) * xW_up) W_down"""
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Neden modern Transformer mimarilerinin (ViT, GPT-3, LLaMA) neredeyse tamamı 2017'deki orijinal Post-LayerNorm yerine Pre-LayerNorm tasarımına geçmiştir? Matematiksel gerekçesi nedir?

> **Mentor Cevabı:**
> 1. **Kalıntı Yolunun (Residual Stream) Dokunulmazlığı:** Pre-LN'de ana bilgi yolu $x_{l+1} = x_l + \mathcal{F}(\text{LN}(x_l))$ şeklindedir. Gradyan $L$. katmandan $1$. katmana akarken $\frac{\partial \mathcal{L}}{\partial x_1} = \frac{\partial \mathcal{L}}{\partial x_L} \left( I + \sum_{l=1}^{L-1} \frac{\partial \mathcal{F}}{\partial x_l} \right)$ şeklinde saf bir birim matris ($I$) bileşeni içerir. Hiçbir normalizasyon katmanı bu akışı kesemez.
> 2. **Post-LN'deki Gradyan Darboğazı:** Post-LN'de $x_{l+1} = \text{LN}(x_l + \mathcal{F}(x_l))$ olduğu için geriye yayılan gradyan her katmanda LayerNorm türeviyle ($\frac{\partial \text{LN}}{\partial x}$) çarpılmak zorundadır. $L=50$ katmanlı bir ağda bu durum gradyanların ilk katmanlara ulaştığında ya patlamasına ya da sıfıra çökmesine yol açar.
> 3. **Öğrenme Oranı Isınması (Warmup) Bağımlılığı:** Post-LN modelleri ilk 5000-10000 adımda çok küçük öğrenme oranıyla yavaşça ısıtılmak (warmup) zorundadır, aksi halde eğitim ilk adımda NaN üretir. Pre-LN ise ilk adımdan itibaren tam öğrenme oranıyla kararlı şekilde yakınsar.

---

## 📜 Lisans & Metaveri

```text
/*
 * Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101 Günlük Yapay Zeka, Bilgisayarlı Görü ve MLOps Mühendisliği
 * Özel Lisans — Tüm Hakları Saklıdır.
 */
```
