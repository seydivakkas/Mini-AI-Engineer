# Day 161: LLaVA Mimarisi — ViT (Vision Transformer) + MLP Projector + LLM ile Uçtan Uca Çok Modlu (VLM) Modeli

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%209-Multimodal%20Foundations-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 9: Çok Modlu (Multimodal) Temel Modeller (Gün 161 - Gün 180)** serisinin açılış projesidir. OpenAI GPT-4V ve LLaVA (Large Language and Vision Assistant) tarzı modern Görsel Dil Modellerinin (Visual Language Models - VLM) omurgasını oluşturan **Vision Transformer (ViT-14x14) Patch Kodlayıcı**, **2 Katmanlı GELU MLP Hizalama Projektörü** ve **Oto-Regresif Dil Modeli (LLM)** mimarisini sıfırdan PyTorch ile hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "LLaVA ve VLM (Visual Language Model)" Nedir ve Bir Metin Modeli Nasıl Görür?
- **Sorun (Metin Modellerinin Körlüğü):**
  Standart bir LLM (örn: LLaMA veya GPT-3) yalnızca metin tokenlarını ($d_{\text{text}}$ boyutunda vektörler) anlar. Bir görüntünün piksel matrisini $(3 \times 224 \times 224)$ doğrudan LLM'e veremezsiniz.
- **Çözüm (LLaVA'nın 3 Aşamalı Görsel Köprüsü):**
  1. **Aşama 1 (ViT Patch Ayrıştırma):** $224 \times 224$ çözünürlüğündeki görüntü, $14 \times 14$ boyutunda $256$ adet küçük kareye (Patch) bölünür. Her kare birer "görsel kelime" (Visual Token) olarak $768$ boyutlu vektöre kodlanır.
  2. **Aşama 2 (MLP Projektör):** Görsel uzay ($768d$) ile dil modelinin metin uzayı ($512d$) birbiriyle uyumsuzdur. 2 katmanlı bir MLP projektör, görsel tokenları metin tokenlarının anlayacağı uzaya yansıtır.
  3. **Aşama 3 (Multimodal Füzyon):** Görsel tokenlar ($256$ adet) ile kullanıcının soru tokenları ($10$ adet) uç uca eklenerek ($266$ token) LLM'e beslenir. LLM resmi adeta bir paragraf metin gibi okur ve cevabı üretir!

```
             LLaVA VLM ARCHITECTURE PIPELINE
  [Görüntü (3, 224, 224)]
           │
           ▼  14x14 Patch Embedding
  [CLIP-ViT-L/14 Vision Encoder]
           │  (256 Patch Token x 768 Boyut)
           ▼
  [2 Katmanlı MLP Projektör (GELU)]
           │  (256 Patch Token x 512 Boyut)
           ▼
  [Multimodal Concatenation: <IMG_TOKENS> + <TEXT>]
           │  (266 Token x 512 Boyut)
           ▼
  [Oto-Regresif LLM (Decoder)] ──> [Metin Yanıtı]
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Vision Transformer (ViT) Patch Ayrıştırma ve Boyut Formülü
- Görüntü boyutu $H \times W = 224 \times 224$, patch boyutu $P = 14$ ise:
  $$N_{\text{patches}} = \left(\frac{H}{P}\right) \times \left(\frac{W}{P}\right) = \left(\frac{224}{14}\right)^2 = 16 \times 16 = 256 \text{ Patch}$$
- Her patch, 2D konvolüsyon (kernel=14, stride=14) ile $d_{\text{vision}} = 768$ boyutuna yansıtılır: $\mathbf{X}_v \in \mathbb{R}^{B \times 256 \times 768}$.

### B. Çapraz Modlu (Cross-Modal) MLP Hizalama Katmanı
- LLaVA-1.5 mimarisinde tek bir doğrusal katman yerine 2 katmanlı GELU MLP kullanılır:
  $$\mathbf{H}_v = \text{GELU}(\mathbf{X}_v \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2 \in \mathbb{R}^{B \times 256 \times d_{\text{text}}}$$
- Bu katman, önceden eğitilmiş dondurulmuş (frozen) ViT ile dondurulmuş LLM arasındaki kavramsal köprüyü kurar.

### C. Multimodal Dizi Sıralaması ve Dikkat Matrisi (Attention)
- Girdi dizisi: $\mathbf{Z} = [\mathbf{H}_v \, ; \, \mathbf{E}_t] \in \mathbb{R}^{B \times (256 + L_{\text{text}}) \times d_{\text{text}}}$
- Kendi kendine dikkat (Self-Attention) mekanizması, soru kelimelerinin ($E_t$) doğrudan ilgili görüntü bölgelerindeki ($H_v$) patch'lere dikkat ağırlığı vermesini sağlar.

### D. İki Aşamalı VLM Eğitim Protokolü
1. **Aşama 1 (Ön-Hizalama / Pre-training):** ViT ve LLM dondurulur, sadece MLP projektör milyonlarca resim-altyazı çiftiyle (CC3M/LAION) eğitilir.
2. **Aşama 2 (Görsel İnce Ayar / Visual SFT):** ViT dondurulur, MLP projektör ve LLM çok modlu diyalog verisiyle (LLaVA-Instruct-150k) uçtan uca ince ayarlanır.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **VLM (Visual Language Model)** | Hem görsel hem de metinsel girdileri eşzamanlı işleyip anlayan çok modlu yapay zeka modeli. |
| **LLaVA** | Large Language and Vision Assistant; açık kaynaklı öncü VLM mimarisi. |
| **ViT (Vision Transformer)** | Görüntüleri patch'lere bölerek Transformer bloklarıyla işleyen görsel kodlayıcı. |
| **Patch Embedding** | 2D görüntü parçasını 1D embedding vektörüne dönüştürme işlemi (örn: 14x14 piksel $\to$ 768d). |
| **Cross-Modal Projector** | Görsel embedding uzayını dil modelinin gizli metin uzayına hizalayan MLP katmanı. |
| **VQA (Visual Question Answering)** | Verilen bir görsel hakkındaki doğal dil sorularını yanıtlama görevi. |
| **Multimodal Fusion** | Görsel ve metinsel tokenların tek bir ortak embedding dizisinde birleştirilmesi. |
| **Prefix Visual Tokens** | Görsel tokenların metin dizisinin başına ön-ek (prefix) olarak eklenmesi stratejisi. |
| **GELU Activation** | Gaussian Error Linear Unit; Transformer ve MLP köprülerinde kullanılan yumuşak aktivasyon. |
| **Spatial Resolution (Grid)** | Görüntünün patch ızgarası çözünürlüğü (örn: $16 \times 16 = 256$ görsel token). |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Mevcut güçlü LLM'leri sıfırdan     │ • 256 görsel token, LLM bağlam       │
 │   eğitmeden görsel yetenekle         │   penceresinde önemli yer kaplar     │
 │   donatma (Eğitim verimliliği).      │   (Bellek ve dikkat maliyeti).       │
 │ • Zengin VQA ve sahne anlama.        │ • Çok küçük nesnelerde 14x14 patch   │
 │                                      │   çözünürlük kaybı yaşanabilir.      │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Medikal görüntü analizi, otonom    │ • Görsel illüzyon ve detaylarda      │
 │   sürüş, robotik algı ve ekran       │   nesne halüsinasyonu (Visual        │
 │   okuma ajanları geliştirme.         │   Hallucination) riski.              │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/vlm_llava_architecture_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
