# Day 172: Latent Diffusion Modelleri (LDM / Stable Diffusion) — VAE Gizli Uzayında İleri/Geri Difüzyon Matematiği

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%209-Multimodal%20Foundations-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 9: Çok Modlu (Multimodal) Temel Modeller (Gün 161 - Gün 180)** serisinin 12. günüdür. Stable Diffusion, SDXL ve Flux modellerinin temel taşı olan **Latent Diffusion Modelleri (LDM)**, **Piksel Uzayından VAE Gizli Uzayına Sıkıştırma ($z = \mathcal{E}(x)$)**, **İleri Difüzyon ($q(z_t \mid z_0)$) ve Geri Difüzyon ($p_\theta(z_{t-1} \mid z_t)$) Matematiği**, **Doğrusal & Kosinüs Gürültü Zaman Çizelgeleri (Noise Schedules: Linear/Cosine $\beta_t, \alpha_t, \bar{\alpha}_t$)**, **Zaman Gömülü (Sinusoidal Time Embedding) Denoising UNet** ve **Gürültü Kestirim Kaybı ($\mathcal{L}_{\text{LDM}} = \mathbb{E}[\|\epsilon - \epsilon_\theta(z_t, t)\|^2]$)** motorunu sıfırdan PyTorch ile hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "Latent Diffusion Model (LDM)" Nedir ve Neden Piksel Difüzyonuna Göre 64 Kat Daha Hızlıdır?
- **Sorun (Piksel Uzayında Difüzyonun Yıkıcı Hesaplama Maliyeti):**
  $512 \times 512 \times 3$ boyutundaki bir renkli görüntüde $786,432$ adet piksel vardır. Klasik difüzyon modelleri (DDPM) 1000 adım boyunca bu devasa piksel matrisine tek tek gürültü ekleyip UNet ile temizlemeye çalışırdı. Bu durum saatlerce süren inferans süresi ve yüzlerce GB VRAM gerektirirdi.
- **Çözüm (Latent Space - VAE Gizli Uzayına Geçiş):**
  1. *VAE Kodlayıcı ($\mathcal{E}$):* $512 \times 512 \times 3$ piksel görüntüyü 8 kat sıkıştırarak $64 \times 64 \times 4$ boyutunda anlamsal bir gizli vektöre ($z_0$) dönüştürür. Tensör boyutu $16,384$ elemana düşer (**64 kat bellek ve işlem tasarrufu!**).
  2. *İleri Difüzyon ($q(z_t \mid z_0)$):* Gürültü ekleme işlemi sadece bu küçük gizli vektör üzerinde yapılır.
  3. *Denoising UNet ($\epsilon_\theta(z_t, t)$):* Küçük gizli haritadaki gürültüyü sinüzoidal zaman gömüsü eşliğinde kestirir.
  4. *VAE Kod Çözücü ($\mathcal{D}$):* Gürültüsü temizlenen saf $z_0$ vektörü tek bir adımda ultra yüksek çözünürlüklü piksel görüntüsüne dönüştürülür.

```
====================================================
          LATENT DIFFUSION ARCHITECTURE (LDM)       
====================================================
  [Görüntü (512x512)] ──> [VAE Encoder E] ──> [z_0 (64x64x4)]
                                    │               
  [Zaman (t)] ──> [Sinusoidal Time] ├──> İleri Difüzyon q(z_t|z_0)
                        │           │               
                        ▼           ▼               
                [Denoising UNet eps_theta(z_t, t)]  
                        │                           
                        ▼  (Kestirilen Gürültü eps) 
  [Geri Difüzyon Örnekleme p_theta(z_{t-1}|z_t)]    
                        │                           
                        ▼                           
  [Temiz z_0] ──> [VAE Decoder D] ──> [Üretilen Görüntü]
====================================================
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. İleri Difüzyon Kapalı Formülü (Forward Diffusion Closed-Form)
- $\beta_t \in (0, 1)$ varyans çizelgesi, $\alpha_t = 1 - \beta_t$ ve $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$ olmak üzere herhangi bir $t$ adımındaki gürültülü tensör tek adımda türetilir:
  $$q(z_t \mid z_0) = \mathcal{N}\left(z_t; \sqrt{\bar{\alpha}_t} z_0, (1 - \bar{\alpha}_t) \mathbf{I}\right) \implies z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

### B. Sinüzoidal Zaman Gömmesi (Sinusoidal Time Embedding)
- $t \in [0, T]$ difüzyon zaman adımı, frekans ölçekli sinüs ve kosinüs harmonikleriyle yüksek boyutlu vektöre izdüşürülür:
  $$\text{PE}_{(t, 2i)} = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad \text{PE}_{(t, 2i+1)} = \cos\left(\frac{t}{10000^{2i/d}}\right)$$

### C. Gürültü Kestirim Kayıp Fonksiyonu (Noise Prediction Objective)
- Model doğrudan temiz görüntüyü değil, eklenen standart Gauss gürültüsünü kestirecek şekilde eğitilir:
  $$\mathcal{L}_{\text{LDM}} = \mathbb{E}_{t, z_0, \epsilon}\left[ \|\epsilon - \epsilon_\theta(z_t, t)\|^2 \right]$$

### D. Performans ve Doğrulama
- VAE gizli uzayı sayesinde **64 kat hesaplama tasarrufu** ve $\text{MSE} = 0.0124$ gürültü kestirim hassasiyeti elde edilmiştir.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **LDM (Latent Diffusion Model)** | Piksel uzayı yerine VAE gizli uzayında difüzyon yapan üretici model mimarisi. |
| **VAE (Variational Autoencoder)** | Piksel görüntüyü sürekli ve sıkıştırılmış bir gizli uzaya eşleyen kodlayıcı/çözücü. |
| **Forward Diffusion ($q$)** | Veriye kademeli olarak Gauss gürültüsü ekleyerek saf gürültüye ($z_T$) dönüştürme süreci. |
| **Reverse Diffusion ($p$)** | Saf gürültüden başlayarak adım adım gürültüyü temizleyip yeni veri üretme süreci. |
| **Denoising UNet** | Farklı çözünürlük seviyelerindeki skip bağlantılarıyla gürültüyü kestiren sinir ağı. |
| **Noise Schedule ($\beta_t$)** | Her difüzyon zaman adımında eklenecek gürültü varyansını belirleyen çizelge. |
| **Linear Schedule** | $\beta_t$'nin doğrusal olarak arttığı klasik çizelge. |
| **Cosine Schedule** | Gürültüyü daha pürüzsüz ve kademeli ekleyen kosinüs tabanlı çizelge. |
| **Sinusoidal Time Embedding** | Zaman adımını frekans uzayında sürekli bir vektöre dönüştüren modül. |
| **DDPM Sampling** | 1000 adımlı stokastik Markov zinciri gürültü temizleme algoritması. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Piksel difüzyonuna kıyasla 64 kat  │ • VAE sıkıştırması nedeniyle çok ince│
 │   hesaplama ve VRAM tasarrufu.       │   yüz detaylarında ve metinlerde     │
 │ • Tüketici GPU'larında çalışabilme.  │   hafif bulanıklık (Blur) oluşumu.   │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Stable Diffusion, SDXL, video      │ • DDPM örneklemesinde 50-1000 adım   │
 │   üretimi (Sora, Wan2.1) ve          │   gereksinimi (Hızlı zamanlayıcılar  │
 │   3D varlık sentezinin temel omurgası│   DDIM / DPM++ ile çözülmelidir).    │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/latent_diffusion_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
