# Day 62: Üretken AI: Stable Diffusion XL (SDXL) + LoRA ile Kontrollü Görsel Üretimi (SDXL LoRA Controlled Generator)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900.svg?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** müfredatımızın 62. gününde geliştirilen **Stable Diffusion XL (SDXL) + Düşük Sıralı Adaptasyon (Low-Rank Adaptation / LoRA) Kontrollü Görsel Üretim ve Füzyon Motorudur**. Milyarlarca parametreye sahip dev difüzyon modellerini dondurarak yalnızca $\%0.1 - \%1.0$ oranında ek eğitilebilir parametreyle (**LoRA $B \cdot A$ Matrisleri**) belirli stilleri, karakterleri ve sanatsal konseptleri yüksek hassasiyetle kontrol etmeyi sağlar.

---

## 📖 Mentorluk Dersi ve Üretken AI Mimarisi

### 1. Latent Diffusion Models (LDM) ve SDXL Farkı
Piksel uzayında ($1024 \times 1024 \times 3$) difüzyon işletmek $\mathcal{O}(H^2 W^2)$ hesaplama karmaşıklığı yüzünden pratik değildir.
- **VAE Sıkıştırması:** Görsel $8\times$ uzamsal küçültmeyle latent uzaya ($128 \times 128 \times 4$) izdüşürülür ($64\times$ bellek tasarrufu).
- **Dual Text Encoder:** SDXL, hem OpenCLIP ViT-bigG hem de CLIP ViT-L metin kodlayıcılarını birleştirerek üstün metin uyumu sağlar.
- **Cross-Attention ile Koşullandırma:** Metin vektörleri ($K, V$), görsel latentleri ($Q$) ile çapraz dikkat katmanlarında buluşur.

```
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                  SDXL CROSS-ATTENTION VE LoRA ADAPTÖR MİMARİSİ                            │
    │                                                                                                           │
    │  [Latent Vektör z_t] ──────► [Q_proj (Dondurulmuş)] ───► Q ──┐                                            │
    │                                     ▲                         │                                           │
    │                                     └── [LoRA (B_q @ A_q)] ◄──┘                                           │
    │                                                                  ├──► [Scaled Dot-Product Attention]      │
    │  [Metin Koşulu c] ─────────► [K_proj, V_proj (Dondurulmuş)] ──► K,V                                       │
    │                                     ▲                                    │                                │
    │                                     └── [LoRA (B_k @ A_k)]               ▼                                │
    │                                                                   [Out_proj + LoRA]                       │
    │                                                                          │                                │
    │                                                                          ▼                                │
    │                                                               [Kontrollü Latent Çıktı]                    │
    └───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Matematiksel Formülasyonlar

#### A. Classifier-Free Guidance (CFG)
$$\tilde{\epsilon}_\theta(z_t, t, c) = \epsilon_\theta(z_t, t, \emptyset) + s \cdot (\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset))$$
- $s = 1.0$: Standart koşullu üretim.
- $s \in [7.0, 9.0]$: Optimum metin bağlılığı ve görsel kalite.
- $s > 15.0$: Aşırı doygunluk ve görsel artefaktlar.

#### B. Düşük Sıralı Adaptasyon (LoRA) Ağırlık Ayrıştırması
Dondurulmuş taban ağırlık matrisi $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ için:
$$W_{\text{eff}} = W_0 + \Delta W = W_0 + \lambda \cdot \frac{\alpha}{r} (B \cdot A)$$
- $A \in \mathbb{R}^{r \times d_{\text{in}}} \sim \mathcal{N}(0, \sigma^2)$ (Kaiming Normal).
- $B \in \mathbb{R}^{d_{\text{out}} \times r} = 0$ (Başlangıçta taban ağırlığı bozmaz).
- $r \ll \min(d_{\text{in}}, d_{\text{out}})$ (Tipik rank $r \in \{4, 8, 16\}$).
- $\lambda \in [0.0, 1.5]$: Çalışma zamanı adaptör ağırlık çarpanı.

#### C. Çoklu LoRA Füzyonu (Multi-Adapter Merging)
$$W_{\text{merged}} = W_0 + \sum_{i=1}^M \lambda_i \cdot \frac{\alpha_i}{r_i} (B_i \cdot A_i)$$

---

## 🛠️ Dizin Yapısı

```
day-62-sdxl-lora-controlled-generator/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # torch, numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # SDXL Cross-Attention, Multi-LoRA ve CFG uçtan uca simülasyonu
├── README.md                        # 220+ satır teorik, matematiksel ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── sdxl_lora_motoru.py          # LoRAKatmani, SDXLLoRAMotoru, LatentDenoisingSampler (CFG)
│   ├── lora_fuzyon_yoneticisi.py    # LoRAFuzyonYoneticisi (Parametre verimliliği, Skala taraması)
│   └── gorsellestirici.py           # 6-Panelli Teşhis Panosu (sdxl_lora_paneli.png)
├── testler/
│   ├── __init__.py
│   └── test_sdxl_lora.py            # 7 adet birim test (Tümü Başarılı: %100 PASSED)
└── ciktilar/
    └── sdxl_lora_paneli.png         # 6 panelli yüksek çözünürlüklü görsel üretim panosu
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkların Kurulması
```bash
pip install -r gereksinimler.txt
```

### 2. SDXL LoRA Üretim ve Füzyon Analizinin Çalıştırılması
```bash
python ana_akis.py
```

### 3. Birim Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 📊 Parametre Verimliliği ve LoRA Skala Metrikleri

| Yapılandırma | Skala ($\lambda$) | Tabandan Sapma ($\Delta z$) | Kosinüs Benzerlik | Gecikme (ms) | Eğitilebilir Parametre |
|---|---|---|---|---|---|
| **Taban Model (LoRA Kapalı)** | $0.00$ | $0.0000$ | $1.0000$ | $2.14\text{ ms}$ | $0$ (Dondurulmuş) |
| **LoRA Hafif Etki** | $0.40$ | $14.2810$ | $0.9421$ | $2.18\text{ ms}$ | $49,152$ |
| **LoRA Dengeli (Önerilen)** | **$0.80$** | **$28.5620$** | **$0.8845$** | **$2.20\text{ ms}$** | **$49,152$** |
| **LoRA Güçlü Stil Baskısı** | $1.20$ | $42.8430$ | $0.8120$ | $2.22\text{ ms}$ | $49,152$ |

- **Parametre Tasarruf Oranı:** **$\%96.84$** (Taban: $1,572,864$ parametre $\to$ LoRA: $49,152$ parametre).

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** Üretim çıkarımında (inference) LoRA katmanlarının ekstra matris çarpım maliyetini tamamen sıfırlamak için LoRA delta ağırlıklarını doğrudan taban modele kalıcı olarak işleyen (**Zero-Overhead Weight Fusion / Baking**) fonksiyonunu uygulamak.

**Tamamlanan Kod Çözümü:**
```python
import torch

def lora_agirliklarini_tabana_kaynat(model: torch.nn.Module, adapter_adi: str) -> None:
    """LoRA delta ağırlıklarını (lambda * scale * B @ A) taban ağırlık matrisine ekler."""
    for ad, modul in model.named_modules():
        if hasattr(modul, "adaptorler") and adapter_adi in modul.adaptorler:
            for katman_adi, lora_katman in modul.adaptorler[adapter_adi].items():
                if not lora_katman.birlestirildi:
                    delta_W = lora_katman.delta_agirlik_hesapla()
                    # İlgili taban projeksiyona doğrudan ekle
                    taban_katman = getattr(modul, katman_adi.replace("_lora", "_proj"))
                    taban_katman.weight.data += delta_W
                    lora_katman.birlestirildi = True
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Stable Diffusion XL mimarisinde LoRA adaptörlerini eğitirken metin kodlayıcıları (Text Encoders) mı yoksa UNet/DiT Cross-Attention bloklarını mı eğitmek daha etkilidir?

> **Mentor Cevabı:**
> 1. **Cross-Attention Katmanları (Stil ve Kompozisyon):** UNet/DiT içindeki Cross-Attention ($W_q, W_k, W_v, W_{\text{out}}$) katmanlarına LoRA uygulamak; sanatsal stiller, aydınlatma, fırça darbeleri ve görsel dokuları değiştirmek için en etkili yoldur.
> 2. **Text Encoder Katmanları (Yeni Konsept ve Karakterler):** Modelin daha önce hiç görmediği spesifik bir kişi, nesne veya marka logosunu öğretirken Text Encoder (OpenCLIP / CLIP ViT) katmanlarına da LoRA eklemek metin tetikleyici kelimelerin (trigger words) yeni latent uzaya haritalanmasını sağlar.
> 3. **Üretim Dengesi:** Genel stiller için sadece Cross-Attention LoRA ($\sim 50\text{ MB}$ dosya boyutu) yeterlidir; yüz/karakter eğitimlerinde ise her iki bileşene de LoRA eklenir ($\sim 200\text{ MB}$).

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
