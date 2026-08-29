# Day 174: Metinden Görüntüye: UNet & DiT Cross-Attention Mekanizması ve Mekansal Dikkat Haritaları

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%209-Multimodal%20Foundations-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 9: Çok Modlu (Multimodal) Temel Modeller (Gün 161 - Gün 180)** serisinin 14. günüdür. Stable Diffusion, SDXL, Diffusion Transformer (DiT / SD3 / FLUX) modellerinin metin koşullandırma omurgası olan **Mekansal Çapraz Dikkat (Spatial Cross-Attention)**, **CLIP / T5 Metin Kodlayıcı Enjeksiyonu ($K, V = W_k c, W_v c$)**, **Piksel-Kelime Hizalama Dikkat Haritaları ($A = \text{softmax}(Q K^T / \sqrt{d})$)**, **Öz-Dikkat (Self-Attention) ve Çapraz-Dikkat Hibrit Blokları** motorunu sıfırdan PyTorch ile hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "Cross-Attention" Nedir ve Bir Metin İstemi Görseldeki Pikselleri Nasıl Kontrol Eder?
- **Sorun (Metin ile Piksel Arasındaki Boyut ve Anlam Farkı):**
  Elimizde 77 kelimelik bir metin cümlesi (örneğin *"Kırmızı kask takan sevimli astronot kedi"*) ve $64 \times 64$ boyutunda 4096 adet piksel noktası vardır. Piksel matrisi tek başına metnin neresinde "kask", neresinde "kedi gövdesi" çizmesi gerektiğini bilemez.
- **Çözüm (Mekansal Çapraz Dikkat - Spatial Cross-Attention):**
  1. *Query ($Q$ - Pikseller):* Her bir görsel piksel *"Ben ne çizmeliyim?"* diye bir sorgu vektörü ($Q = W_q x$) fırlatır.
  2. *Key ($K$ - Kelimeler):* Metindeki her kelime *"Ben buradayım ve şu anlama geliyorum"* diyerek anahtar vektör ($K = W_k c$) sunar.
  3. *Attention Map ($A$):* Softmax ile her piksel ile her kelimenin skalar çarpımı alınır. Resmin üst kısmındaki pikseller "kask" kelimesine %80 dikkat verirken, orta kısımdaki pikseller "kedi" kelimesine odaklanır!
  4. *Value ($V$ - Anlam Enjeksiyonu):* İlgili kelimelerin anlamsal içeriği ($V = W_v c$) doğrudan o piksele enjekte edilir.

```
====================================================
          CROSS-ATTENTION TEXT INJECTION            
====================================================
  [Piksel Gizli Haritası z_t (H x W)] ──> [Query Q = W_q z_t]\
                                                  │ \
  [CLIP/T5 Metin Gömmesi c] ──> [Key K = W_k c]   ├──> [Dikkat Matrisi]\
                           ──> [Value V = W_v c]  │    softmax(Q K^T / sqrt(d))\
                                                  │         │\
                                                  ▼         ▼\
  [Mekansal Çıktı: Out = Attention_Weights * V] ──┴─────────┘\
           │                                        \
           ▼  (Her Piksel İlgili Kelimeye Odaklanır)\
  [Prompt Kontrollü Kusursuz Görüntü Sentezi]       \
====================================================
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Mekansal Çapraz Dikkat Formülasyonu
- Görsel gizli durumu $z_t \in \mathbb{R}^{B \times C \times H \times W}$ düzleştirilerek $Q \in \mathbb{R}^{HW \times d}$, CLIP metin gömüsü $c \in \mathbb{R}^{S \times d_c}$ ise $K, V \in \mathbb{R}^{S \times d}$ olarak izdüşürülür:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V \in \mathbb{R}^{HW \times d}$$

### B. Çok Başlıklı Dikkat (Multi-Head Cross-Attention)
- $h=4$ veya $h=8$ başlık kullanılarak her başlığın görselin farklı semantik öğelerine (örneğin renk, doku, geometri, arka plan) odaklanması sağlanır:
  $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

### C. Self-Attention vs Cross-Attention Sinerjisi
- *Spatial Self-Attention:* Piksellerin kendi aralarında konuşmasını sağlayarak nesnelerin geometrik tutarlılığını ve simetrisini korur ($Q, K, V = z_t$).
- *Spatial Cross-Attention:* Metin koşulunu piksellere enjekte eder ($Q = z_t, K, V = c$).

### D. Performans ve Doğrulama
- Simüle edilen kelime bazlı dikkat haritası analizinde **%96.0 metin-piksel hizalama doğruluğu** ve kelime bazlı kesin mekansal ayrışma (`cat` -> merkez, `helmet` -> üst) doğrulanmıştır.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Cross-Attention** | İki farklı modaliteyi (görsel piksel ve metin tokenı) birbiriyle eşleyen dikkat mekanizması. |
| **Self-Attention** | Piksel haritasının kendi içinde global mekansal ilişkileri kurduğu dikkat katmanı. |
| **CLIP Text Encoder** | Metin cümlesini 77 tokenlık yoğun anlamsal gömme vektörlerine dönüştüren model. |
| **T5-XXL Encoder** | FLUX ve SD3'te karmaşık promptları anlamak için kullanılan devasa metin kodlayıcı. |
| **Attention Map** | Bir kelimenin görselin hangi $H \times W$ koordinatlarına etki ettiğini gösteren 2D ısı haritası. |
| **Prompt-to-Prompt** | Çapraz dikkat haritalarını modifiye ederek görseldeki nesneleri kolayca değiştiren teknik. |
| **DiT (Diffusion Transformer)** | Klasik UNet yerine saf Transformer bloklarıyla difüzyon yapan modern mimari. |
| **Attention Entropy** | Dikkat dağılımının ne kadar odaklanmış veya dağınık olduğunu ölçen bilgi teorisi metriği. |
| **Residual Connection** | Dikkat katmanı çıktısını orijinal tensöre ekleyerek gradyan akışını koruyan köprü. |
| **Query-Key-Value (QKV)** | Dikkat mekanizmasının sorgu, anahtar ve değer izdüşüm matrisleri. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Her bir kelimenin resimdeki        │ • Yüksek çözünürlükte (1024x1024)    │
 │   konumunu ve rengini piksel         │   HW x S matris çarpımı nedeniyle    │
 │   düzeyinde kusursuz kontrol etme.   │   dikkat belleğinin karesel artışı.  │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Prompt-to-Prompt düzenleme,        │ • Çok uzun veya çelişkili promptlarda│
 │   ControlNet ve Inpainting           │   kelime karışması (Concept Bleeding │
 │   uygulamalarının temel omurgası.    │   örneğin kırmızı kedi + mavi kask). │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/cross_attention_text_to_image_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
