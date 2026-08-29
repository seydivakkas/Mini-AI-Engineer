# Day 177: Diffusion Transformers (DiT - Sora, SD3 ve Flux Omurgası)

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%209-Multimodal%20Foundations-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 9: Çok Modlu (Multimodal) Temel Modeller (Gün 161 - Gün 180)** serisinin 17. günüdür. OpenAI Sora, Black Forest Labs Flux ve Stable Diffusion 3 mimarilerinin temel omurgasını oluşturan **Diffusion Transformers (DiT - Peebles & Xie, 2023)**, **Patchify Mekansal Yama Ayrıştırma (p=2, 4)**, **adaLN-Zero (Adaptive Layer Normalization: $\gamma, \beta, \alpha$ regresyonu ve sıfır-başlatma)**, **Zaman ve Metin Koşullandırma Projeksiyonu ($t + c$)**, **Unpatchify ve Lineer Rekonstrüksiyon**, ve **Ölçeklenebilirlik (Scaling Laws: DiT-S, DiT-B, DiT-L, DiT-XL)** motorunu sıfırdan PyTorch ile hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ Neden UNet Terk Edildi ve OpenAI Sora / Flux "DiT" Mimarisine Geçti?
- **Sorun (Konvolüsyonel UNet'lerin Tıkanması):**
  Geleneksel Stable Diffusion 1.5 / 2.1 modellerinde UNet (konvolüsyon ağları) kullanılırdı. Ancak konvolüsyon katmanları *"İndüktif Önyargı"* (Inductive Bias) taşır; yani sadece yakın piksellere bakar. Modeli daha da büyüttüğünüzde (Compute artışı) UNet bir noktadan sonra daha zeki hale gelmez ve öğrenmesi doyuma ulaşır.
- **Çözüm (Saf Transformer + Patchify + adaLN-Zero):**
  1. *Patchify (Görseli Token Yapma):* 2D görsel, tıpkı metin cümlelerindeki kelimeler gibi $2 \times 2$ piksellik küçük karelere (yamalara) bölünür ve 1D token dizisi yapılır.
  2. *Saf Self-Attention:* Her yama, görseldeki diğer tüm yamalarla aynı anda konuşur (Küresel dikkat).
  3. *adaLN-Zero Modülasyonu:* Zaman adımı ($t$) ve metin istemi ($c$), Transformer bloklarının içine `LayerNorm` katmanlarını ölçekleyip kaydırarak (`gamma, beta, alpha`) enjekte edilir. Başlangıçta sıfır başlatıldığı için model ilk adımda kimlik fonksiyonu gibi pürüzsüz çalışır.
  - *Devrim:* Hesaplama gücü (FLOPs) arttıkça görsel üretim kalitesi **doğrusal olarak iyileşir (Scaling Laws)**!

```
====================================================
         adaLN-ZERO DIFFUSION TRANSFORMER BLOCK     
====================================================
  Koşul c (t + Metin) ──> [Linear (6xD)] ───────────┐
                                                    │
  Görsel Tokenları (x) ────────────────────────┐    │ (gamma, beta, alpha)
           │                                   │    │
           ▼                                   ▼    ▼
  [LayerNorm] ──> [Modulate (gamma1, beta1)] ──> [Multi-Head Self-Attn]
                                                        │
  x ──────────────── (+) ◄── [Scale by alpha1_gate] ────┘
           │
           ▼
  [LayerNorm] ──> [Modulate (gamma2, beta2)] ──> [Feed-Forward MLP]
                                                        │
  x ──────────────── (+) ◄── [Scale by alpha2_gate] ────┘
           │
           ▼
  [Sonraki DiT Bloğuna Aktar: x_out]
====================================================
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Patchify ve Token Sayısı Matematiği
- $H \times W \times C$ boyutundaki VAE gizli tensörü, $p \times p$ yama boyutuyla düzleştirilerek token dizisine dönüştürülür:
  $$N = \frac{H}{p} \times \frac{W}{p}, \quad D = p^2 \cdot C \xrightarrow{\text{Linear}} d_{\text{model}}$$
- $p=2$ seçimi, $p=4$ veya $p=8$'e kıyasla $4 \times$ daha fazla token ($N$) üreterek piksel detay sadakatini zirveye taşır.

### B. adaLN-Zero Modülasyonu (Adaptive Layer Normalization with Zero Initialization)
- $c = \text{MLP}(t) + \text{TextEmb}$ koşul vektöründen 6 adet parametre üretilir:
  $$[\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2] = \text{Linear}(c)$$
- Blok içi modülasyon formülü:
  $$\text{Modulate}(x, \gamma, \beta) = x \odot (1 + \gamma) + \beta$$
  $$x \leftarrow x + \alpha_1 \cdot \text{SelfAttention}(\text{Modulate}(\text{LN}(x), \gamma_1, \beta_1))$$
  $$x \leftarrow x + \alpha_2 \cdot \text{MLP}(\text{Modulate}(\text{LN}(x), \gamma_2, \beta_2))$$

### C. DiT Ölçeklenme Yasaları (Scaling Laws)
- Parametre sayısı (Small 33M -> X-Large 675M) ve hesaplama (GFLOPs) arttıkça Fréchet Inception Distance (FID) $10.50$'den **$2.27$'ye düşmektedir**.

### D. Performans ve Doğrulama
- 8/8 PyTest testi ve 6 panelli teşhis panosu ile patchify, adaLN modülasyonu ve gradyan akışı doğrulanmıştır.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **DiT (Diffusion Transformer)** | UNet yerine tamamen standart Transformer bloklarıyla çalışan difüzyon mimarisi. |
| **Patchify** | 2D uzaysal görüntüyü küçük yama bloklarına bölerek 1D token dizisi haline getirme işlemi. |
| **Unpatchify** | İşlenmiş token dizisini tekrar 2D uzaysal görüntü tensörüne dönüştüren işlem. |
| **adaLN-Zero** | Transformer LayerNorm parametrelerini zaman ve metinle modüle eden ve sıfır başlatılan katman. |
| **Scaling Laws** | Model parametresi ve hesaplama gücü arttıkça model performansının tahmin edilebilir şekilde artması. |
| **Inductive Bias** | Bir mimarinin veriye dair varsayımları (Konvolüsyonun yerel piksel varsayımı gibi). |
| **Self-Attention** | Görseldeki tüm yamaların birbirleriyle olan anlamsal ilişkisini hesaplayan dikkat mekanizması. |
| **Positional Embedding** | Yamaların görsel üzerindeki 2D koordinat bilgisini (x, y) temsil eden vektör. |
| **FID (Fréchet Inception Distance)** | Üretilen görsellerin gerçek görsellere olan kalite ve çeşitlilik benzerlik metriği (Düşük = İyi). |
| **GFLOPs** | Modelin tek bir ileri beslemede yaptığı milyar kayan nokta işlemi sayısı. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Konvolüsyonel önyargısız küresel   │ • Küçük yama boyutlarında (p=2)      │
 │   dikkat (Global Self-Attention).    │   token sayısının ve karesel bellek  │
 │ • Donanım ölçeklenebilirliği (Sora). │   maliyetinin ($O(N^2)$) artması.    │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Video difüzyonu (3D spatio-temporal│ • Eğitim için yüzlerce H100 GPU      │
 │   yama tokenları), fotogerçekçi      │   ve devasa veri kümelerine          │
 │   yüksek çözünürlüklü görüntü sentezi│   ihtiyaç duyması.                   │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/dit_diffusion_transformers_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
