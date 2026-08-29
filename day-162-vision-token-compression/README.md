# Day 162: Görüntü Token Sıkıştırma — BLIP-2 Q-Former, C-Abstractor ve Spatial Pooling

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%209-Multimodal%20Foundations-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 9: Çok Modlu (Multimodal) Temel Modeller (Gün 161 - Gün 180)** serisinin 2. günüdür. Vision Transformer (ViT) tarafından üretilen 256-1024 adet ham görsel tokenın LLM'in dikkat ($O(N^2)$) ve bellek maliyetini tıkamasını engellemek amacıyla endüstride kullanılan 3 büyük sıkıştırma mimarisini (**BLIP-2 Q-Former**, **C-Abstractor** ve **Spatial Pooling 2x2/4x4**) sıfırdan PyTorch ile inşa edip kıyaslamaktadır.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "Görüntü Token Sıkıştırma (Vision Token Compression)" Neden Hayati Önemdedir?
- **Sorun (Görsel Token Patlaması ve Dikkat Boğulması):**
  Bir görsel dil modeline (VLM) yüksek çözünürlüklü bir görüntü (örn: $448 \times 448$ veya çoklu resim) verdiğinizde, ViT tek bir resim için $1024$ adet token üretir. 4 resimli bir diyalogda sadece resimler $4096$ token kaplar!
  LLM'in dikkat hesaplaması token sayısının karesiyle ($O(N^2)$) büyüdüğü için sistem yavaşlar ve bellek taşar.
- **Çözüm (3 Farklı Sıkıştırma Felsefesi):**
  1. **BLIP-2 Q-Former (%87.5 Sıkıştırma):** $32$ adet öğrenilebilir akıllı sorgu tokenı (Query) görsel sahneye bakar ve sadece soruyla/anlamla ilgili öznitelikleri çeker. $256$ token $\to 32$ token!
  2. **C-Abstractor (%75 Sıkıştırma):** 2D derinlikli konvolüsyon (Depthwise Conv 3x3) ile komşu pikselleri birleştirip yerel kenar/doku detaylarını korur. $256$ token $\to 64$ token.
  3. **Spatial Pooling (Parametresiz & Hızlı):** Adaptif $2 \times 2$ havuzlama ile doğrudan $256 \to 64$ tokena indirger.

```
+---------------------------------------------------------------------------------------------+
|                            VISION TOKEN COMPRESSION SUMMARY                                 |
|  1. Ham ViT (Sıkıştırmasız)    : 256 Token | %0.0 Sıkıştırma   | %0.0 Bellek Tasarrufu      |
|  2. Spatial Pooling (2x2)      : 64 Token  | %75.0 Sıkıştırma  | %93.8 Bellek Tasarrufu     |
|  3. C-Abstractor (Conv 2x)     : 64 Token  | %75.0 Sıkıştırma  | %93.8 Bellek Tasarrufu     |
|  4. BLIP-2 Q-Former (32 Query) : 32 Token  | %87.5 Sıkıştırma  | %98.4 Bellek Tasarrufu! 🏆 |
+---------------------------------------------------------------------------------------------+
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. BLIP-2 Q-Former Çapraz Dikkat (Cross-Attention) Mekanizması
- $N_q = 32$ adet öğrenilebilir parametre $\mathbf{Q} \in \mathbb{R}^{B \times 32 \times d}$, ViT çıktıları $\mathbf{X}_v \in \mathbb{R}^{B \times 256 \times d}$ üzerinde anahtar ve değer olarak kullanılır:
  $$\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{W}_Q (\mathbf{X}_v \mathbf{W}_K)^T}{\sqrt{d_k}}\right) (\mathbf{X}_v \mathbf{W}_V)$$
- Bu sayede 256 dağınık görsel patch, en bilgilendirici 32 vektörde özetlenir.

### B. C-Abstractor'ın Konvolüsyonel İndirgeme ve Receptive Field Üstünlüğü
- 1D Sequence halindeki tokenlar tekrar 2D ızgaraya $(B, C, 16, 16)$ dönüştürülür ve $3 \times 3$ Derinlikli Konvolüsyon (Depthwise Conv, stride=2) uygulanır:
  $$H_{\text{out}} = \left\lfloor \frac{16 + 2(1) - 3}{2} + 1 \right\rfloor = 8 \implies 8 \times 8 = 64 \text{ Token}$$
- C-Abstractor, piksellerin uzamsal komşuluk ilişkisini koruyarak OCR ve küçük nesne algılama başarısını artırır.

### C. Self-Attention $O(N^2)$ Bellek ve FLOPs Tasarruf Analizi
- LLM içindeki dikkat matrisi belleği token sayısının karesiyle orantılıdır:
  $$\text{Bellek Tasarrufu} = 1 - \left(\frac{N_{\text{compressed}}}{N_{\text{raw}}}\right)^2 = 1 - \left(\frac{32}{256}\right)^2 = 1 - \frac{1}{64} = 98.4375\%$$
- 32 token kullanımı, LLM'in dikkat katmanında **%98.4 bellek ve FLOPs tasarrufu** sağlar!

### D. Bilgi Sıkıştırma Boğazı (Information Bottleneck) Dengesi
- Aşırı sıkıştırma ($N_q < 16$), sahnedeki ince detayların ve metinlerin (OCR) kaybolmasına yol açabilir.
- $N_q = 32-64$ aralığı, hem VQA hem de genel sahne betimlemede Pareto optimal çalışma noktasıdır.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Q-Former** | Querying Transformer; öğrenilebilir sorgularla görsel tokenları damıtan mimari. |
| **C-Abstractor** | Convolutional Abstractor; 2D konvolüsyon ile uzamsal görsel tokenları sıkıştıran modül. |
| **Spatial Pooling** | 2D ızgarada komşu görsel patch tokenlarını ortalama/maksimum havuzlama ile birleştirme. |
| **Information Bottleneck** | Girdi bilgisini gereksiz gürültüden arındırıp en kritik anlamsal özeti tutma prensibi. |
| **Cross-Attention** | Sorguların (Query) bir kaynaktan, Anahtar/Değerlerin (Key/Value) başka bir kaynaktan geldiği dikkat türü. |
| **Learnable Queries** | Eğitim sırasında güncellenen ve sahneden bilgi çeken rastgele başlatılmış tensörler. |
| **Depthwise Conv** | Her kanalın bağımsız 2D filtre ile taranarak parametre verimliliği sağlayan konvolüsyon. |
| **Attention Memory Footprint** | $O(N^2)$ büyüklüğündeki dikkat matrislerinin GPU VRAM'de kapladığı alan. |
| **Resampler** | Flamingo modelinde kullanılan Q-Former benzeri görsel token indirgeme mekanizması. |
| **Token Pruning** | Anlamsal önemi düşük olan görsel tokenların dikkat skorlarına göre elenmesi. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • LLM bağlam penceresinde %98.4'e    │ • Çok küçük yazılarda (OCR) veya     │
 │   varan dikkat bellek tasarrufu.     │   mikro nesnelerde hafif detay       │
 │ • Çoklu resim ve video işlemeyi      │   kaybı riski.                       │
 │   mümkün kılan kompakt token boyu.   │                                      │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Video-LLM'ler, yüksek çözünürlüklü │ • Eğitilebilir Q-Former katmanının   │
 │   doküman analizi ve mobil cihazlar  │   ön-eğitim veri setine bağımlılığı  │
 │   için hafif VLM dağıtımı.           │   ve optimizasyon zorluğu.           │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/vision_token_compression_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
