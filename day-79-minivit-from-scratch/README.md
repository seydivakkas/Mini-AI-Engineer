# Day 79: Sıfırdan Mini Vision Transformer (MiniViT) — Patch Projeksiyonu, CLS Token, Encoder Birleşimi

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](gereksinimler.txt)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Model: MiniViT](https://img.shields.io/badge/Architecture-Mini_Vision_Transformer-orange.svg?style=flat-square)](#matematiksel-formülasyon)
[![Tests: 8/8 Passed](https://img.shields.io/badge/pytest-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/test_minivit.py)

Dosovitskiy et al. (2020) *"An Image is Worth 16x16 Words"* makalesindeki Vision Transformer (ViT) mimarisini; **Görsel Yama Gömülme (Patch Embedding)**, **Öğrenilebilir [CLS] Token**, **1D Pozisyonel Gömülmeler (Positional Embeddings)**, **Çok Katmanlı Pre-LN Transformer Encoder Yığını** ve **Attention Rollout Görselleştiricisi** ile sıfırdan saf PyTorch kullanarak inşa ediyoruz.

---

## 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)

Bilgisayarlı Görü alanında 2012'den beri egemen olan Evrişimli Sinir Ağları (CNN), görsel pikselleri lokal pencerelerle ($3 \times 3$ filtreler) işler. Ancak Vision Transformer (ViT) şu temel bilimsel gerekçelerle görsel işlemede yeni bir çağ açmıştır:

1. **Evrişimsiz Saf Dizi Modellemesi (Pure Sequence Paradigm):**
   2D görsel $P \times P$ boyutunda küçük kare yamalara (ör. $4 \times 4$ veya $16 \times 16$) bölünür ve düzleştirilerek tıpkı NLP'deki kelime token'ları gibi bir dizi haline getirilir. Görsel ve metin artık tek bir standart Transformer omurgasında birleşebilir (Multimodal Uyumluluk).
2. **İlk Katmandan İtibaren Küresel Alıcı Alan (Instant Global Receptive Field):**
   CNN'lerin görselin bir ucundaki nesne ile diğer ucundaki arka planı ilişkilendirmesi için onlarca pooling ve evrişim katmanından geçmesi gerekirken, ViT **1. katmandan itibaren tüm yamalar arasında $O(1)$ doğrudan dikkat bağı kurar.**
3. **[CLS] Token ile Merkezi Semantik Temsil:**
   Tüm yama dizisinin en başına özel ve öğrenilebilir bir `[CLS]` (Classification) token'ı eklenir. Dikkat katmanları boyunca bu token tüm yamalardan özet bilgi toplar; çıkışta sadece bu token'ın temsili MLP kafasına verilerek sınıflandırma yapılır.
4. **Attention Rollout ile Kusursuz Açıklanabilirlik (Explainable AI):**
   Katmanlar arasındaki dikkat matrisleri birleştirilerek modelin görselin tam olarak hangi bölgelerine odaklandığı (Attention Rollout) 2D ısı haritası olarak doğrudan görselleştirilebilir.

---

## 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)

- **Lokal Alıcı Alan Kısıtı (Local Receptive Field Bottleneck):**
  CNN'ler büyük nesneleri veya uzamsal ilişkileri kavramakta gecikir. ViT ilk katmandan itibaren küresel dikkat uygular.
- **Evrişim Katmanlarının Donanım Verimsizliği:**
  Evrişim işlemleri im2col veya özel CUDA kernel optimizasyonları gerektirirken, ViT tamamen standart GEMM (General Matrix Multiply) matris çarpımlarına dayanır; GPU ve TPU tensör çekirdeklerini %100 doldurur.
- **Multimodal Entegrasyon Engeli:**
  Görsel için CNN, metin için Transformer kullanmak yerine; hem görsel hem metin artık aynı Transformer Encoder bloklarında tek parça olarak işlenebilir.

---

## ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)

- **Doğal Tümevarımsal Yanlılık (Inductive Bias) Eksikliği:**
  CNN'ler piksellerin uzamsal komşuluğunu ve öteleme değişmezliğini (Translation Equivariance) doğuştan bilir. ViT ise bu uzamsal düzeni sıfırdan öğrenmek zorundadır; bu yüzden **küçük veri setlerinde CNN'ler kadar hızlı yakınsayamaz, güçlü regülarizasyon (Mixup, CutMix) ve ön eğitim gerektirir.**
- **Yüksek Çözünürlükte Karesel Bellek Artışı:**
  Görsel boyutu $1024 \times 1024$ olduğunda yama sayısı $N = 4096$ olur ve dikkat matrisi $4096 \times 4096$ belleği zorlar. *(Çözüm: Swin Window Attention veya Patch Size büyütme).*

---

## 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar

| Mimari | Girdi Temsili | Alıcı Alan | Ön Eğitim İhtiyacı | Doğruluk Potansiyeli |
|---|---|---|---|---|
| **Mini Vision Transformer (Bizim Model)** | **$4 \times 4$ Yamalar + [CLS]** | **Küresel ($O(1)$)** | **Yüksek / Orta** | ⭐⭐⭐⭐⭐ (SOTA) |
| **ResNet-50 (Standart CNN)** | $7 \times 7$ Evrişim + Pooling | Kademeli Büyüyen | Düşük (ImageNet) | ⭐⭐⭐⭐ |
| **Swin Transformer** | Kayan Pencereli Yamalar | Hiyerarşik / Lokal Pencere | Orta | ⭐⭐⭐⭐⭐ |
| **ConvNeXt (Modernize CNN)** | $7 \times 7$ Derinlikli Evrişim | Genişletilmiş Lokal | Düşük / Orta | ⭐⭐⭐⭐⭐ |

---

## 📐 Matematiksel Formülasyon

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                MINI VISION TRANSFORMER (MiniViT) MİMARİSİ                                 │
│                                                                                                           │
│   Görsel x ∈ ℝ^(3 × 32 × 32)                                                                              │
│       │                                                                                                   │
│       ▼ (Conv2D: kernel=4, stride=4, D=64)                                                                │
│   64 Yama Gömülmesi: x_p ∈ ℝ^(64 × 64)                                                                    │
│       │                                                                                                   │
│       ▼ (Öğrenilebilir [CLS] Token Başa Eklenir: [CLS; x_p])                                              │
│   65 Token Dizisi: z_0 ∈ ℝ^(65 × 64)                                                                      │
│       │                                                                                                   │
│       ▼ (+ Öğrenilebilir Pozisyonel Gömülme E_pos)                                                        │
│   z_0 = z_0 + E_pos                                                                                       │
│       │                                                                                                   │
│       ▼ (L=4 Katmanlı Pre-LN Transformer Encoder Yığını)                                                  │
│   z_L = EncoderStack(z_0) ∈ ℝ^(65 × 64)                                                                   │
│       │                                                                                                   │
│       ▼ (Sadece [CLS] Token Temsili Alınır: z_L[0])                                                       │
│   y_hat = Linear( LayerNorm( z_L[0] ) ) ──> 10 Sınıf Logiti                                              │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Yama Gömülme Projeksiyonu (Patch Embedding)
Girdi görseli $x \in \mathbb{R}^{C \times H \times W}$, $P \times P$ boyutunda $N = \frac{H \cdot W}{P^2}$ adet yamaya bölünür ve $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ matrisiyle projekte edilir:

$$x_p = [x_p^1 E; x_p^2 E; \dots; x_p^N E] \in \mathbb{R}^{N \times D}$$

### 2. [CLS] Token ve Pozisyonel Kodlama Entegrasyonu
Öğrenilebilir $x_{\text{class}} \in \mathbb{R}^{1 \times D}$ ve $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ ile:

$$z_0 = [x_{\text{class}}; x_p^1 E; x_p^2 E; \dots; x_p^N E] + E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$$

### 3. Çok Katmanlı Transformer Encoder Yığını
$$z_\ell' = \text{MHSA}(\text{LayerNorm}(z_{\ell-1})) + z_{\ell-1}, \quad z_\ell = \text{FFN}(\text{LayerNorm}(z_\ell')) + z_\ell', \quad \ell = 1 \dots L$$

### 4. Sınıflandırma Kafası (MLP Head)
$$y = \text{Linear}\big(\text{LayerNorm}(z_L^0)\big) \in \mathbb{R}^{C_{\text{classes}}}$$

### 5. Attention Rollout Formülasyonu
Katmanlar arası kümülatif dikkat akışı (Abnar & Zuidema 2020):

$$A_{\text{rollout}} = \prod_{\ell=1}^L \left( 0.5 \cdot \bar{A}_\ell + 0.5 \cdot \mathbf{I} \right)$$

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama |
|---|---|---|
| **Patch Embedding** | *Yama Gömülmesi* | 2D görseli örtüşmeyen $P \times P$ parçalara bölüp doğrusal projeksiyonla $D$ boyutlu vektörlere dönüştürme işlemi. |
| **[CLS] Token** | *Classification Token* | Tüm görsel yamalarından dikkat mekanizmasıyla küresel semantik bilgi toplayan öğrenilebilir temsil vektörü. |
| **Attention Rollout** | *Dikkat Akışı Yayılımı* | Tüm Transformer katmanlarındaki dikkat matrislerini birleştirerek [CLS] token'ın görseldeki odak alanlarını 2D haritalama yöntemi. |
| **Inductive Bias** | *Tümevarımsal Yanlılık* | Bir modelin mimarisine gömülü ön kabuller (CNN'deki öteleme değişmezliği ve yerellik gibi). |
| **MLP Head** | *Sınıflandırma Kafası* | [CLS] token temsilini nihai sınıf logitlerine dönüştüren LayerNorm ve Linear katmanı. |
| **Patch Resolution ($P$)**| *Yama Çözünürlüğü* | Her görsel yamasının piksel boyutu ($4 \times 4$, $8 \times 8$, $16 \times 16$). $P$ küçüldükçe yama sayısı ve hesaplama artar. |
| **Truncated Normal Init** | *Kırpılmış Normal Başlatma* | [CLS] ve $E_{\text{pos}}$ parametrelerini $\pm 2\sigma$ aralığında rastgele başlatarak gradyan patlamasını önleyen yöntem. |

---

## 📊 SWOT Analizi ile Karar Matrisi

| Kategori | Açıklama ve Endüstriyel Karar Kriteri |
|---|---|
| **Strengths (Güçlü Yönler)** | İlk katmandan itibaren küresel alıcı alan; [CLS] token ile yüksek temsil gücü; Attention Rollout ile açıklanabilirlik. |
| **Weaknesses (Zayıf Yönler)** | Küçük veri setlerinde CNN'ler kadar hızlı tümevarımsal genelleme yapamaz; Güçlü veri artırma zorunluluğu. |
| **Opportunities (Fırsatlar)** | Masked Autoencoder (MAE) ve DINO ile self-supervised ön eğitim; LoRA ile düşük maliyetli fine-tuning. |
| **Threats (Tehditler)** | Yetersiz regülarizasyon ve küçük batch boyutlarında modelin yerel minimumlara takılması. |

---

## 💻 Üretim Seviyesinde Uygulama Mimarisi

Tam kaynak kodları [`day-79-minivit-from-scratch/`](.) dizinindedir:

```python
class MiniVisionTransformer(nn.Module):
    def __init__(self, gorsel_boyutu=32, yama_boyutu=4, giris_kanali=3, sinif_sayisi=10, gomulme_boyutu=64, derinlik=4, kafa_sayisi=4):
        super().__init__()
        self.patch_embed = YamaGomulmeKatmani(gorsel_boyutu, yama_boyutu, giris_kanali, gomulme_boyutu)
        num_patches = self.patch_embed.toplam_yama_sayisi

        self.cls_token = nn.Parameter(torch.zeros(1, 1, gomulme_boyutu))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, gomulme_boyutu))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.bloklar = nn.ModuleList([
            TransformerEncoderBlogu(model_boyutu=gomulme_boyutu, kafa_sayisi=kafa_sayisi)
            for _ in range(derinlik)
        ])
        self.norm = OzelLayerNorm(gomulme_boyutu)
        self.head = nn.Linear(gomulme_boyutu, sinif_sayisi)

    def forward(self, x, dikkat_haritalarini_don=False):
        b = x.shape[0]
        x_patches = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x_patches), dim=1) + self.pos_embed

        dikkat_listesi = []
        for blok in self.bloklar:
            x, att = blok(x)
            dikkat_listesi.append(att)

        cls_temsili = self.norm(x)[:, 0]
        logitler = self.head(cls_temsili)
        return (logitler, dikkat_listesi) if dikkat_haritalarini_don else logitler
```

---

## 📊 Model Parametre Dağılımı ve Doğrulama Çıktıları

`ana_akis.py` çalıştırılarak elde edilen analitik parametre dökümü:

```text
=================================================================
           Bileşen             | Parametre Sayısı |   Oran (%)  
=================================================================
Yama Gömülme (Conv2D)          |      3,136       | %   1.5    
CLS & Pozisyonel Gömülme       |      4,224       | %   2.0    
Encoder Blokları (MHSA+FFN)    |     199,936      | %  96.1    
Norm & MLP Head                |        778       | %   0.4    
-----------------------------------------------------------------
TOPLAM MİNİVİT KAPASİTESİ      |     208,074      | % 100.0
=================================================================
```

### 🔑 Analiz ve Çıkarımlar
1. **Dengeli Model Ölçeği:** Toplam $208.074$ parametre ile model, CIFAR-10 gibi $32 \times 32$ veri setlerinde sıfırdan eğitilebilecek kadar hafif, ancak Transformer dinamiklerini tam yansıtacak kadar güçlüdür.
2. **Kusursuz Gradyan Akışı:** Geriye yayılımda [CLS] token ($2.1814$), Positional Embedding ($2.1870$) ve Patch Embedding ($1.4452$) katmanlarının tamamı sağlıklı gradyanlar almıştır.
3. **Birim Test Güvencesi:** [`testler/test_minivit.py`](testler/test_minivit.py) altındaki **8/8 birim test %100 PASSED (3.84s)**.

---

## 🎨 6 Panelli Teşhis Panosu

Üretilen yüksek çözünürlüklü teşhis paneli [`ciktilar/minivit_mimari_paneli.png`](ciktilar/minivit_mimari_paneli.png) konumundadır:

1. **MiniViT Uçtan Uca Hesaplama Akışı:** Patch Embedding'den MLP Head'e tam mimari akış.
2. **Görselin 4x4 Yamalara Ayrıştırılması:** $32 \times 32$ görselin $N=64$ adet yama ızgarası görseli.
3. **[CLS] Token Attention Rollout:** Modelin görselde odaklandığı bölgelerin yarı saydam ısı haritası.
4. **Pozisyonel Gömülmelerin Kosinüs Benzerliği:** $65 \times 65$ pozisyon korelasyon matrisi.
5. **MiniViT Parametre Dağılımı:** Pasta grafiğiyle bileşen bazında ağırlık oranları.
6. **MiniViT Mimari SWOT Karar Matrisi:** Mühendislik karar tablosu.

---

## 🧪 Günün Alıştırması & Zorlu Görevi

**Görev:** [CLS] token kullanmak yerine, tüm görsel yamalarının çıkış temsilleri üzerinde **Küresel Ortalama Havuzlama (Global Average Pooling - GAP)** uygulayan ve bunu isteğe bağlı kılan hibrit bir MiniViT sınıflandırma kafası yazınız.

```python
import torch
import torch.nn as nn

class HibritViTSiniflandirici(nn.Module):
    """[CLS] token veya Global Average Pooling (GAP) destekli sınıflandırıcı."""
    def __init__(self, d_model: int = 64, sinif_sayisi: int = 10, havuzlama_tipi: str = "cls"):
        super().__init__()
        self.havuzlama_tipi = havuzlama_tipi.lower()
        assert self.havuzlama_tipi in ["cls", "gap"], "havuzlama_tipi 'cls' veya 'gap' olmalıdır."
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, sinif_sayisi)

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        # x_tokens: (Batch, 1 + N, D)
        if self.havuzlama_tipi == "cls":
            # Sadece 0. indeks [CLS] token'ı al
            temsil = x_tokens[:, 0]
        else:
            # 1. indeksten itibaren tüm görsel yamalarının ortalamasını al
            temsil = x_tokens[:, 1:].mean(dim=1)
        
        return self.head(self.norm(temsil))
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Vision Transformer mimarisinde neden tüm görsel yamalarının ortalamasını almak (Global Average Pooling) yerine başa fazladan bir `[CLS]` token'ı eklenmiştir? Bu iki yaklaşım arasındaki temsil farkı nedir?

> **Mentor Cevabı:**
> 1. **Önyargısız Bilgi Toplama (Bias-Free Aggregation):** Eğer doğrudan görsel yamalarının ortalaması (GAP) alınırsa, görselin arka planındaki anlamsız yamalar (boş gökyüzü, beyaz duvar) ile asıl nesnenin bulunduğu yamalar eşit ağırlıkla toplanır. `[CLS]` token ise öğrenilebilir bir sorgu ($Q$) gibi davranarak Self-Attention üzerinden sadece kritik yamalardan yüksek ağırlıklı bilgi çeker.
> 2. **NLP & Multimodal Uyumluluk (BERT Standartı):** BERT ve GPT gibi dil modellerinde sınıflandırma ve cümle düzeyi temsil için `[CLS]` token standarttır. ViT'in de `[CLS]` kullanması, görsel ve metin token'larının CLIP veya VLM modellerinde aynı potada eritilmesini kolaylaştırır.
> 3. **Attention Rollout Kolaylığı:** Sınıflandırma kararının doğrudan `[CLS]` token üzerinden verilmesi, bu token'ın diğer yamalara olan dikkat katsayılarının izlenerek modelin kararının açıklanmasını (Explainability) son derece basit ve zarif kılar.

---

## 📜 Lisans & Metaveri

```text
/*
 * Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
 * 101 Günlük Yapay Zeka, Bilgisayarlı Görü ve MLOps Mühendisliği
 * Özel Lisans — Tüm Hakları Saklıdır.
 */
```
