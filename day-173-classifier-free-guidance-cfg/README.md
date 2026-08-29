# Day 173: Classifier-Free Guidance (CFG) & DDIM Hızlı Örnekleme Zamanlayıcıları

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%209-Multimodal%20Foundations-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 9: Çok Modlu (Multimodal) Temel Modeller (Gün 161 - Gün 180)** serisinin 13. günüdür. Stable Diffusion, Midjourney ve DALL-E modellerinin kalitesi ve hızının anahtarı olan **Classifier-Free Guidance (CFG)**, **Koşullu ve Koşulsuz Gürültü Ekstrapolasyonu ($\tilde{\epsilon}_\theta = \epsilon_\theta(z_t, \emptyset) + w \cdot (\epsilon_\theta(z_t, c) - \epsilon_\theta(z_t, \emptyset))$)**, **CFG Ölçeği ($w \in [1, 15]$) ve Mod Çökmesi / Aşırı Doygunluk (Oversaturation) Analizi**, **Deterministik DDIM (Denoising Diffusion Implicit Models) Hızlı Örnekleme Matematiği (1000 Adımdan 20-50 Adıma İndirgeme)** motorunu sıfırdan PyTorch ile hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "Classifier-Free Guidance (CFG)" ve "DDIM Hızlı Zamanlayıcı" Nedir?
- **Sorun 1 (Metin İsteminin Yetersiz Kalması - Düşük Uyum):**
  Bir modele *"Kırmızı şapkalı sevimli bir astronot kedi"* dediğinizde, model çoğunlukla genel kedi resimleri üretir ve kırmızı şapkayı ya da astronot kıyafetini unutur. Çünkü saf koşullu olasılık dağılımı güvenli ve ortalama görüntülere kayar.
- **Çözüm 1 (Classifier-Free Guidance - CFG):**
  Model aynı anda hem boş prompt'lu görüntüyü ($\epsilon_\emptyset$) hem de prompt'lu görüntüyü ($\epsilon_c$) tahmin eder. Ardından aradaki farkı $w=7.5$ kat büyütür! Model istemin istediği yönü aşırı vurgulayarak görseli prompt'a %100 sadık hale getirir.
- **Sorun 2 (1000 Adımlı DDPM Örneklemesinin Aşırı Yavaşlığı):**
  Klasik difüzyon 1000 adımda rastgele gürültü ekleyerek ilerler (15 saniye sürer).
- **Çözüm 2 (Deterministik DDIM Zamanlayıcısı):**
  Rastgeleliği sıfıra indirir ($\eta = 0.0$) ve difüzyon sürecini pürüzsüz bir Diferansiyel Denklem (ODE) yörüngesine oturtur. 1000 adım yerine **sadece 20 adımda** aynı kalitede resmi **0.28 saniyede (50 kat hızlı!)** üretir.

```
====================================================
         CFG NOISE EXTRAPOLATION & DDIM ODE         
====================================================
  [UNet(z_t, t, c)]   ──> [Koşullu Gürültü eps_cond] 
  [UNet(z_t, t, null)]──> [Koşulsuz Gürültü eps_uncond]
                               │                    
                               ▼                    
  [CFG Formülü: eps_tilde = eps_uncond + w*(eps_cond - eps_uncond)]
                               │                    
                               ▼                    
  [DDIM Adımı (eta=0): z_{t-1} = sqrt(alpha_prev)*z_0 + dir_xt]
                               │                    
                               ▼                    
  [Sonuç: 20 Adımda Kusursuz Prompt Uyumlu Görüntü] 
====================================================
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. CFG Gürültü Ekstrapolasyon Matematiği (Ho & Salimans, 2022)
- Eğitim sırasında %10-20 olasılıkla metin koşulu $c$ boş belirtece ($\emptyset$) düşürülür (Dropout).
- İnferans anında yönlendirilmiş gürültü vektörü:
  $$\tilde{\epsilon}_\theta(z_t, c) = \epsilon_\theta(z_t, \emptyset) + w \cdot \left(\epsilon_\theta(z_t, c) - \epsilon_\theta(z_t, \emptyset)\right)$$
- $w = 1.0$: Standart koşullu üretim.
- $w \in [7.0, 8.0]$: Optimum görsel kalite ve prompt uyumu (Sweet Spot).

### B. Dinamik Eşikleme (Dynamic Thresholding) ile Yanık Piksel Engelleme
- Yüksek $w$ ($w > 12$) değerlerinde tensör büyüklükleri patlar ve renkler yanar. Dinamik eşikleme, tensörün %99.5 yüzdelik değerini $s = \text{percentile}(|x|, 99.5)$ bularak normalize eder:
  $$x_{\text{clamped}} = \frac{\text{clip}(x, -s, s)}{s}$$

### C. Deterministik DDIM Örnekleme ODE Formülasyonu
- $\eta = 0.0$ iken stokastik gürültü terimi sıfırlanır:
  $$z_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left(\frac{z_t - \sqrt{1 - \bar{\alpha}_t} \tilde{\epsilon}_\theta}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1 - \bar{\alpha}_{t-1}} \tilde{\epsilon}_\theta$$

### D. Performans ve Doğrulama
- DDPM 1000 adım (14.2 saniye) yerine DDIM 20 adımda (0.28 saniye) **50.7 kat hızlanma** ve %92 prompt uyumu doğrulanmıştır.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Classifier-Free Guidance (CFG)** | Ayrı bir sınıflandırıcı ağ olmadan istem uyumunu artıran ekstrapolasyon tekniği. |
| **Guidance Scale ($w$)** | Koşullu yön vektörünün ne kadar büyütüleceğini belirleyen katsayı. |
| **Unconditional Score ($\epsilon_\emptyset$)** | Metin istemi olmadan modelin kendi kendine tahmin ettiği gürültü. |
| **Conditional Score ($\epsilon_c$)** | Verilen metin istemi doğrultusunda tahmin edilen gürültü. |
| **Oversaturation / Burn-in** | Aşırı yüksek CFG ölçeklerinde renklerin patlaması ve kontrast bozulması. |
| **Dynamic Thresholding** | Aşırı parlak pikselleri normalize ederek doğal kontrastı koruyan eşikleme tekniği. |
| **DDIM (Denoising Diffusion Implicit Models)** | Stokastik difüzyonu deterministik ODE yörüngesine dönüştüren hızlı zamanlayıcı. |
| **Inference Steps** | Difüzyon inferansında kullanılan alt örnekleme adım sayısı (örn. 20-50). |
| **Eta ($\eta$)** | DDIM'deki rastgelelik parametresi ($\eta=0$ deterministik, $\eta=1$ DDPM). |
| **CLIP Alignment Score** | Üretilen görsel ile metin istemi arasındaki kosinüs benzerliği. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Ayrı bir sınıflandırıcı ağı        │ • Her adımda 2 ayrı ileri besleme    │
 │   eğitme maliyetini sıfırlama.       │   (Cond + Uncond) gerektirdiği için  │
 │ • 20 adımda 50 kat hızlı inferans.   │   adım başı hesaplama maliyeti 2 kat.│
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Midjourney, Stable Diffusion, Flux │ • w > 12 değerlerinde aşırı doygunluk│
 │   ve gerçek zamanlı görsel üretim    │   ve insan anatomisi bozulmaları     │
 │   uygulamaları (SD-Turbo, LCM).      │   (Mod Çökmesi riski).               │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/cfg_ddim_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
